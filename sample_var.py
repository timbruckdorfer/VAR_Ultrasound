"""
VAR + VQVAE sweep sampler (multi-scale).
Inputs (CLI):
- --out_dir:  directory containing VAR checkpoints (expects ar-ckpt-best.pth or ar-ckpt-last.pth)
- --tok_dir:  directory with train/tokens_multiscale_maps.npz and classes.json
- --vae_ckpt: VQ-VAE checkpoint (.pth)
- --codebook: codebook weights (.pth)
- sampling knobs (temperature/top-k/top-p, repeat penalty, teacher prefix mode, etc.)

Outputs:
- For each (scale boundary K, teacher-prefix tag TP|NOP):
  - PNG samples per class under <out_dir>/samples_sweep3/...
  - tokens_gen.npz with: idx (N, L), label (N,), vnums (S,)
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torchvision.utils import save_image

from models.vqvae import VQVAE
from models.var   import VAR


# -------------------- basic helpers --------------------
@torch.no_grad()
def _decode_with_fallback(vae, token_ids, vnums):
    """Decode tokens to images using decode_tokens if available, else idxBl_to_img."""
    dev = next(vae.parameters()).device
    token_ids = token_ids.to(dev).long()
    lens = [v * v for v in vnums]
    if hasattr(vae, "decode_tokens") and callable(getattr(vae, "decode_tokens")):
        return vae.decode_tokens(token_ids).clamp(0, 1)
    parts = list(torch.split(token_ids, lens, dim=1))
    return vae.idxBl_to_img(parts, same_shape=True, last_one=True).clamp(0, 1)

def _get_vnums_from_npz(npz_path, default=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)):
    d = np.load(npz_path, allow_pickle=False)
    if "vnums" in d:
        return tuple(int(x) for x in d["vnums"].tolist())
    return tuple(default)

def _lens_from_vnums(vnums: Tuple[int, ...]) -> List[int]:
    return [int(v) * int(v) for v in vnums]

def _seg_bounds(lens: List[int]) -> np.ndarray:
    return np.cumsum([0] + list(lens))  # length S+1

def _lin(a, b, s, S):  # s in [0, S-1]
    return a + (b - a) * (s / max(1, S - 1))


# -------------------- stats from NPZ --------------------
@torch.no_grad()
def build_sampling_stats_from_npz(npz_path: Path, vocab: int, device: str):
    """
    Build sampling stats from tokens_multiscale_maps.npz:
      - pos_prior:   [C, Ltot, V]
      - teacher_pref:[C, Ltot]  (first sample per class)
      - uni_bias:    [V]
      - vnums, lens
    """
    d = np.load(npz_path, allow_pickle=False)
    vnums = tuple(int(x) for x in d["vnums"].tolist()) if "vnums" in d else (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    lens  = _lens_from_vnums(vnums)
    Ltot  = int(sum(lens))
    labels = d["label"].astype(np.int64)
    C = int(np.max(labels)) + 1
    V = int(vocab)

    # Stitch idx1d [N, Ltot]
    N = d[f"tok_{vnums[0]}x{vnums[0]}"].shape[0]
    idx1d = np.empty((N, Ltot), dtype=np.int64)
    for i in range(N):
        off = 0
        for v in vnums:
            tok = d[f"tok_{v}x{v}"][i].reshape(-1)
            idx1d[i, off:off + v*v] = tok
            off += v*v

    # Position-wise class-conditional prior
    counts = np.full((C, Ltot, V), 0.5, dtype=np.float64)  # light smoothing
    for seq, c in zip(idx1d, labels):
        c = int(c)
        for p in range(Ltot):
            counts[c, p, int(seq[p])] += 1.0
    counts /= counts.sum(axis=-1, keepdims=True)
    pos_prior = torch.from_numpy(counts).float().to(device)

    # Teacher prefix: first sample per class (zeros if absent)
    teacher = np.zeros((C, Ltot), dtype=np.int64)
    for c in range(C):
        pos = np.where(labels == c)[0]
        teacher[c] = idx1d[pos[0]] if len(pos) > 0 else np.zeros((Ltot,), np.int64)
    teacher = torch.from_numpy(teacher).long().to(device)

    # Unigram bias
    binc = np.bincount(idx1d.reshape(-1), minlength=V).astype(np.float64)
    uni_prob = binc / max(1.0, binc.sum())
    UNIGRAM_ALPHA = 0.3
    uni_bias = torch.from_numpy(UNIGRAM_ALPHA * np.log(uni_prob + 1e-8)).float().to(device)

    return pos_prior, teacher, uni_bias, vnums, lens

@torch.no_grad()
def build_teacher_bank(npz_path: Path, device: str, max_per_class: int = 0):
    """
    Build a teacher-bank (multiple real idx1d per class) for random prefixes.
    max_per_class=0 uses all samples of that class.
    Returns: bank(list of [Mi, Ltot] LongTensor), vnums, lens
    """
    d = np.load(npz_path, allow_pickle=False)
    vnums = tuple(int(x) for x in d["vnums"].tolist()) if "vnums" in d else (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    lens  = _lens_from_vnums(vnums)
    Ltot  = int(sum(lens))
    labels = d["label"].astype(np.int64)
    C = int(labels.max()) + 1

    def one_idx(i):
        off = 0
        out = np.empty((Ltot,), np.int64)
        for v in vnums:
            tok = d[f"tok_{v}x{v}"][i].reshape(-1)
            out[off:off+v*v] = tok
            off += v*v
        return out

    ids_all = [np.where(labels == c)[0] for c in range(C)]
    bank = []
    for ids in ids_all:
        if max_per_class > 0:
            ids = ids[:max_per_class]
        if len(ids) == 0:
            bank.append(torch.zeros(1, Ltot, dtype=torch.long, device=device))
            continue
        arr = np.stack([one_idx(i) for i in ids], axis=0)
        bank.append(torch.from_numpy(arr).long().to(device))
    return bank, vnums, lens


# -------------------- load models --------------------
def load_models_for_sampling(out_dir: Path, vae_ckpt: Path, codebook_path: Path,
                             vocab: int, vnums: Tuple[int, ...],
                             depth: int, dim: int, heads: int, num_classes: int, device: str):
    # VAE
    vae = VQVAE(vocab_size=vocab, z_channels=16, ch=64, beta=0.25,
                test_mode=True, v_patch_nums=vnums).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=False)
    with torch.no_grad():
        vae.quantize.embedding.weight.data.copy_(torch.load(codebook_path, map_location="cpu"))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # VAR
    var = VAR(vae_local=vae, num_classes=num_classes, depth=depth,
              embed_dim=dim, num_heads=heads, patch_nums=vnums,
              drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
              flash_if_available=False, fused_if_available=False).to(device)
    if hasattr(var, "init_weights"):
        var.init_weights()

    ckpt_best = out_dir / "ar-ckpt-best.pth"
    ckpt_last = out_dir / "ar-ckpt-last.pth"
    ckpt = ckpt_best if ckpt_best.exists() else ckpt_last
    assert ckpt.exists(), f"Checkpoint not found: {ckpt_best} or {ckpt_last}"
    var.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    var.eval()
    return var, vae


# -------------------- core sampler --------------------
@torch.no_grad()
def sample_one_class(
    var, vae, cid: int, num_samples: int,
    pos_prior: torch.Tensor, teacher_pref: torch.Tensor, uni_bias: torch.Tensor,
    vnums: Tuple[int, ...],
    temp_min=0.95, temp_max=0.80, topk_min=64, topk_max=128, topp_min=0.90, topp_max=0.95,
    coarse_K: int = 255, teacher_mode: str = "random",  # "fixed" | "random" | "none"
    use_coarse_argmax: bool = True,
    repeat_penalty: float = 1.20, repeat_window: int = 64,
    device: str = "cuda",
    teacher_bank: Optional[List[torch.Tensor]] = None
):
    """Generate token ids of shape (num_samples, L_total) for one class."""
    lens = _lens_from_vnums(vnums)
    L_total = int(sum(lens))
    S = len(lens)
    seg = _seg_bounds(lens)
    def span(si: int): return int(seg[si]), int(seg[si+1])

    B = int(num_samples)
    lab = torch.full((B,), int(cid), device=device, dtype=torch.long)
    y_tokens = torch.full((B, L_total), -1, device=device, dtype=torch.long)

    K = min(int(coarse_K), L_total)
    if teacher_mode == "fixed":
        y_tokens[:, :K] = teacher_pref[cid].unsqueeze(0).expand(B, -1)[:, :K]
    elif teacher_mode == "random":
        assert teacher_bank is not None and len(teacher_bank[cid]) > 0, "empty teacher bank"
        pool = teacher_bank[cid]  # [M, Ltot]
        sel  = torch.randint(0, pool.size(0), (B,), device=device)
        y_tokens[:, :K] = pool[sel, :K]
    else:  # "none"
        for p in range(K):
            dist = pos_prior[cid, p].unsqueeze(0).expand(B, -1)
            tok  = dist.argmax(-1, keepdim=True) if use_coarse_argmax else torch.multinomial(dist, 1)
            y_tokens[:, p] = tok.squeeze(1)

    for s_idx in range(S):
        st, ed = span(s_idx)
        if ed <= K:
            continue

        # Visible tokens → VAR input
        y_now = y_tokens.clone(); y_now[:, ed:] = -1
        parts = list(torch.split(torch.clamp(y_now, min=0), lens, dim=1))
        x_cvae = vae.quantize.idxBl_to_var_input(parts)  # [B, L, C]
        logits_full = var(lab, x_cvae)
        Lp = logits_full.size(1)
        j0 = max(0, min(st, Lp - 1))
        j1 = max(0, min(ed, Lp))

        temp = float(_lin(temp_min, temp_max, s_idx, S))
        topk = int(round(_lin(topk_min, topk_max, s_idx, S)))
        topp = float(_lin(topp_min, topp_max, s_idx, S))

        for pos in range(max(j0, K), j1):
            logits_t = logits_full[:, pos, :] - uni_bias.view(1, -1)

            # Repeat-penalty within a window
            if repeat_penalty and repeat_penalty > 1.0:
                window = y_tokens[:, max(0, pos - repeat_window):pos]
                for b in range(B):
                    uniq = torch.unique(window[b][window[b] >= 0])
                    if len(uniq) > 0:
                        logits_t[b, uniq] /= repeat_penalty

            # Top-k
            if topk > 0:
                k = min(topk, logits_t.size(-1))
                kth = torch.topk(logits_t, k=k, dim=-1).values[..., -1, None]
                logits_t = torch.where(logits_t < kth, torch.full_like(logits_t, float("-inf")), logits_t)

            # Top-p
            if 0.0 < topp < 1.0:
                sorted_logits, idx = torch.sort(logits_t, descending=True, dim=-1)
                probs_sorted = torch.softmax(sorted_logits / max(1e-6, temp), dim=-1)
                keep = probs_sorted.cumsum(-1) <= topp
                keep[..., 0] = True
                sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
                logits_t = torch.full_like(logits_t, float("-inf")).scatter(-1, idx, sorted_logits)

            # Temperature sampling
            tok = torch.multinomial(torch.softmax(logits_t / max(1e-6, temp), dim=-1), 1).squeeze(1)
            y_tokens[:, pos] = tok

    return y_tokens  # (B, L_total)


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Sweep sampling for VAR+VQVAE across scales and teacher-prefix settings.")
    ap.add_argument("--out_dir",   required=True, help="Dir with ar-ckpt-best/last")
    ap.add_argument("--tok_dir",   required=True, help="Dir containing train/tokens_multiscale_maps.npz and classes.json")
    ap.add_argument("--vae_ckpt",  required=True, help="VQ-VAE checkpoint (.pth)")
    ap.add_argument("--codebook",  required=True, help="Codebook weights (.pth)")
    ap.add_argument("--per_class", type=int, default=4, help="#samples per class per combo")
    ap.add_argument("--depth",     type=int, default=6)
    ap.add_argument("--dim",       type=int, default=384)
    ap.add_argument("--heads",     type=int, default=6)
    ap.add_argument("--vocab",     type=int, default=2048)
    ap.add_argument("--seed",      type=int, default=0)

    # Sampling strategy
    ap.add_argument("--temp_min", type=float, default=0.95)
    ap.add_argument("--temp_max", type=float, default=0.80)
    ap.add_argument("--topk_min", type=int,   default=64)
    ap.add_argument("--topk_max", type=int,   default=128)
    ap.add_argument("--topp_min", type=float, default=0.90)
    ap.add_argument("--topp_max", type=float, default=0.95)
    ap.add_argument("--repeat_penalty", type=float, default=1.20)
    ap.add_argument("--repeat_window",  type=int,   default=64)

    # Teacher prefix mode
    ap.add_argument("--tp_mode", choices=["fixed", "random", "none"], default="random",
                    help="Teacher prefix: fixed=class-first, random=per-image from bank, none=disabled")
    ap.add_argument("--tp_bank_max", type=int, default=0,
                    help="Max real samples per class in teacher bank (0=all; used with --tp_mode random)")

    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    sweep_root = out_dir / "samples_sweep3"
    sweep_root.mkdir(parents=True, exist_ok=True)

    tok_train = Path(args.tok_dir) / "train" / "tokens_multiscale_maps.npz"
    cls_json  = Path(args.tok_dir) / "train" / "classes.json"
    assert tok_train.exists(), f"Missing {tok_train}"
    assert cls_json.exists(),  f"Missing {cls_json}"

    classes = json.load(open(cls_json))
    num_classes = len(classes)

    # vnums / lens / stats
    vnums = _get_vnums_from_npz(tok_train)
    lens  = _lens_from_vnums(vnums)
    seg   = _seg_bounds(lens)
    # 10 COARSE_K values: cumulative length per scale (skip 0)
    coarse_K_list = [int(seg[i]) for i in range(1, len(seg))]  # len == 10

    pos_prior, teacher_pref, uni_bias, vnums_npz, lens_npz = build_sampling_stats_from_npz(tok_train, args.vocab, device)
    assert tuple(vnums_npz) == tuple(vnums), "vnums mismatch between NPZ and computed values"

    teacher_bank = None
    if args.tp_mode == "random":
        teacher_bank, vnums_tb, lens_tb = build_teacher_bank(tok_train, device, max_per_class=args.tp_bank_max)
        assert tuple(vnums_tb) == tuple(vnums), "vnums mismatch between teacher bank and stats"

    # load models
    var, vae = load_models_for_sampling(out_dir, Path(args.vae_ckpt), Path(args.codebook),
                                        args.vocab, vnums, args.depth, args.dim, args.heads, num_classes, device)

    total_saved = 0
    for i, K in enumerate(coarse_K_list, start=1):
        for use_teacher in (True, False):
            tag = "TP" if use_teacher else "NOP"
            teacher_mode = args.tp_mode if use_teacher else "none"
            comb_dir = sweep_root / f"scale_{i:02d}_K{K:05d}" / tag
            comb_dir.mkdir(parents=True, exist_ok=True)

            gen_idx_list = []
            gen_lab_list = []

            for cid, cname in enumerate(classes):
                save_dir = comb_dir / f"class_{cid:02d}_{cname}"
                save_dir.mkdir(parents=True, exist_ok=True)

                ids = sample_one_class(
                    var, vae, cid, args.per_class,
                    pos_prior, teacher_pref, uni_bias,
                    vnums,
                    temp_min=args.temp_min, temp_max=args.temp_max,
                    topk_min=args.topk_min, topk_max=args.topk_max,
                    topp_min=args.topp_min, topp_max=args.topp_max,
                    coarse_K=K, teacher_mode=teacher_mode, use_coarse_argmax=True,
                    repeat_penalty=args.repeat_penalty, repeat_window=args.repeat_window,
                    device=device, teacher_bank=teacher_bank
                )
                imgs = _decode_with_fallback(vae, ids, vnums)
                for n, img in enumerate(imgs):
                    save_image(img, str(save_dir / f"img_{n:02d}.png"))
                    total_saved += 1

                gen_idx_list.append(ids.detach().cpu().numpy())
                gen_lab_list.append(np.full((ids.size(0),), cid, dtype=np.int64))

            # Save tokens for this combination
            gen_idx = np.concatenate(gen_idx_list, axis=0)
            gen_lab = np.concatenate(gen_lab_list, axis=0)
            np.savez_compressed(
                comb_dir / "tokens_gen.npz",
                idx=gen_idx, label=gen_lab,
                vnums=np.asarray(vnums, dtype=np.int64)
            )

            print(f"[{i:02d}/10] K={K} | {tag} → saved {gen_idx.shape[0]} samples.")

    print(f"Done: saved {total_saved} images to {sweep_root}")
    expected = 10 * 2 * args.per_class * num_classes
    if total_saved != expected:
        print(f"Warning: expected {expected}, but saved {total_saved}. Verify class count/paths.")

if __name__ == "__main__":
    main()
