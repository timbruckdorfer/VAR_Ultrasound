#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sampling for VAR from 'sampling_pack' (fixed teacher prefix, GREEDY decode)
- 无任何数据增强（不加载数据集、不用 torchvision transforms）
- 固定前缀（teacher_pref），K 逐尺度累加（10 个尺度）
- 每类生成 N 张（默认 10），完全“去随机”：每个 token 直接取 argmax
- 输出目录默认：{VAR_DIR}/samples_S10xC12xN10_greedyTP
- 可选保存 tokens_gen.npz（--save_tokens）

用法：
python sample_var_greedy.py \
  --var_dir /path/to/VAR_online_tokens_fix21 \
  --vae_ckpt /path/to/vae/best.pth \
  --codebook /path/to/codebook_weight.pth \
  --per_class 10 \
  --sweep_tag samples_S10xC12xN10_greedyTP \
  --seed 0 \
  --save_tokens
"""

import argparse, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torchvision.utils import save_image

# ---------------- Dist safety patch (do NOT edit quant.py) ----------------
import torch.distributed as _dist
if not hasattr(_dist, "_orig_get_world_size"):
    _dist._orig_get_world_size = _dist.get_world_size
def _safe_get_world_size(*args, **kwargs):
    try:
        if _dist.is_available() and _dist.is_initialized():
            return _dist._orig_get_world_size(*args, **kwargs)
    except Exception:
        pass
    return 1
_dist.get_world_size = _safe_get_world_size
# -------------------------------------------------------------------------

# project models
from models.vqvae import VQVAE
from models.var   import VAR


# ---------------- helpers ----------------
def _lens_from_vnums(vnums: Tuple[int, ...]) -> List[int]:
    return [int(v)*int(v) for v in vnums]

def _seg_bounds(lens: List[int]) -> np.ndarray:
    return np.cumsum([0] + list(lens))  # length S+1

@torch.no_grad()
def _decode_tokens(vae: VQVAE, token_ids: torch.LongTensor, vnums: Tuple[int, ...]) -> torch.Tensor:
    """token_ids: [B, Ltot] → images [B, C, H, W] in [0,1]"""
    dev = next(vae.parameters()).device
    token_ids = token_ids.to(dev).long()
    lens = _lens_from_vnums(vnums)
    if hasattr(vae, "decode_tokens") and callable(getattr(vae, "decode_tokens")):
        return vae.decode_tokens(token_ids).clamp(0, 1)
    parts = list(torch.split(token_ids, lens, dim=1))
    return vae.idxBl_to_img(parts, same_shape=True, last_one=True).clamp(0, 1)


# ---------------- sampling_pack ----------------
def load_sampling_pack(var_dir: Path):
    pack = var_dir / "sampling_pack"
    assert pack.exists(), f"sampling_pack not found under: {pack}"

    classes = json.loads((pack / "classes.json").read_text())
    vnums   = tuple(json.loads((pack / "vnums.json").read_text()))
    lens    = _lens_from_vnums(vnums)
    Ltot    = int(sum(lens))

    stats = np.load(pack / "stats_full.npz", allow_pickle=False)
    pos_prior    = torch.from_numpy(stats["pos_prior"]).float()     # [C, Ltot, V]
    teacher_pref = torch.from_numpy(stats["teacher_pref"]).long()   # [C, Ltot]
    uni_bias     = torch.from_numpy(stats["uni_bias"]).float()      # [V]
    V            = int(stats["pos_prior"].shape[-1])

    # teacher_bank 与 greedy/fixed 无关，这里忽略
    return classes, vnums, lens, Ltot, V, pos_prior, teacher_pref, uni_bias


# ---------------- models ----------------
def load_models(var_dir: Path, vae_ckpt: Path, codebook: Path,
                vocab: int, vnums: Tuple[int, ...],
                depth: int, dim: int, heads: int,
                num_classes: int, device: str):
    # VQVAE
    vae = VQVAE(vocab_size=vocab, z_channels=16, ch=64, beta=0.25,
                test_mode=True, v_patch_nums=vnums).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=False)
    with torch.no_grad():
        vae.quantize.embedding.weight.data.copy_(torch.load(codebook, map_location="cpu"))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # VAR
    var = VAR(
        vae_local=vae, num_classes=num_classes,
        depth=depth, embed_dim=dim, num_heads=heads,
        patch_nums=vnums,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        flash_if_available=False, fused_if_available=False
    ).to(device)
    if hasattr(var, "init_weights"):
        var.init_weights()

    ckpt_best = var_dir / "ar-ckpt-best.pth"
    ckpt_last = var_dir / "ar-ckpt-last.pth"
    ckpt = ckpt_best if ckpt_best.exists() else ckpt_last
    assert ckpt.exists(), f"找不到 VAR 权重：{ckpt_best} 或 {ckpt_last}"
    var.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    var.eval()
    for p in var.parameters():
        p.requires_grad_(False)

    return var, vae


# ---------------- GREEDY core ----------------
@torch.no_grad()
def sample_one_class_greedy(
    var, vae, cid: int, num_samples: int,
    teacher_pref: torch.Tensor, uni_bias: torch.Tensor,
    vnums: Tuple[int, ...],
    coarse_K: int,
    device: str = "cuda",
):
    """
    固定 teacher 前缀 + 贪心解码（逐位 argmax），无随机项
    Returns: [B, Ltot] token ids
    """
    lens = _lens_from_vnums(vnums)
    L_total = int(sum(lens))
    S = len(lens)
    seg = _seg_bounds(lens)
    def span(si: int): return int(seg[si]), int(seg[si+1])

    B = int(num_samples)
    lab = torch.full((B,), int(cid), device=device, dtype=torch.long)
    y_tokens = torch.full((B, L_total), -1, device=device, dtype=torch.long)

    K = min(int(coarse_K), L_total)
    # 固定 teacher 前缀（每张一样）
    y_tokens[:, :K] = teacher_pref[cid].to(device).unsqueeze(0).expand(B, -1)[:, :K]

    # 逐尺度、逐位置贪心
    for s_idx in range(S):
        st, ed = span(s_idx)
        if ed <= K:
            continue

        y_now = y_tokens.clone()
        y_now[:, ed:] = -1
        parts = list(torch.split(torch.clamp(y_now, min=0), lens, dim=1))
        x_cvae = vae.quantize.idxBl_to_var_input(parts)  # [B, L, C]
        logits_full = var(lab, x_cvae)                   # [B, L, V]
        Lp = logits_full.size(1)
        j0 = max(0, min(st, Lp - 1))
        j1 = max(0, min(ed, Lp))

        for pos in range(max(j0, K), j1):
            # 减去 unigram 偏置后直接 argmax
            logits_t = logits_full[:, pos, :].to(device) - uni_bias.to(device).view(1, -1)
            tok = torch.argmax(logits_t, dim=-1)  # [B]
            y_tokens[:, pos] = tok

    return y_tokens


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var_dir",   required=True, help="包含 ar-ckpt-best/last 与 sampling_pack 的训练输出目录")
    ap.add_argument("--vae_ckpt",  required=True, help="VQVAE best.pth")
    ap.add_argument("--codebook",  required=True, help="codebook_weight.pth")
    ap.add_argument("--per_class", type=int, default=10, help="每类生成张数（默认10）")
    ap.add_argument("--depth",     type=int, default=6)
    ap.add_argument("--dim",       type=int, default=384)
    ap.add_argument("--heads",     type=int, default=6)
    ap.add_argument("--vocab",     type=int, default=2048)
    ap.add_argument("--seed",      type=int, default=0)
    # 改了默认目录名，避免和之前重复
    ap.add_argument("--sweep_tag", type=str, default="samples_S10xC12xN10_greedyTP", help="输出子目录名")
    ap.add_argument("--save_tokens", action="store_true", help="每个 scale 保存 tokens_gen.npz")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    var_dir = Path(args.var_dir)
    sweep_root = var_dir / args.sweep_tag
    sweep_root.mkdir(parents=True, exist_ok=True)

    classes, vnums, lens, Ltot, V, _pos_prior, teacher_pref, uni_bias = load_sampling_pack(var_dir)
    num_classes = len(classes)
    var, vae = load_models(var_dir, Path(args.vae_ckpt), Path(args.codebook),
                           args.vocab, vnums, args.depth, args.dim, args.heads, num_classes, device)

    # 10 个尺度对应的 K
    seg = _seg_bounds(lens)
    coarse_K_list = [int(seg[i]) for i in range(1, len(seg))]

    manifest = {
        "var_dir": str(var_dir),
        "sweep_tag": args.sweep_tag,
        "classes": classes,
        "vnums": list(map(int, vnums)),
        "L_total": int(Ltot),
        "per_class": int(args.per_class),
        "prefix_mode": "fixed",
        "decode": "greedy_argmax",
        "scales": [{"idx": i+1, "K": int(K)} for i, K in enumerate(coarse_K_list)],
        "items": []
    }

    total_saved = 0
    for i, K in enumerate(coarse_K_list, start=1):
        comb_dir = sweep_root / f"scale_{i:02d}_K{K:05d}"
        comb_dir.mkdir(parents=True, exist_ok=True)

        all_idx_np = []
        all_lab_np = []

        for cid, cname in enumerate(classes):
            save_dir = comb_dir / f"class_{cid:02d}_{cname}"
            save_dir.mkdir(parents=True, exist_ok=True)

            ids = sample_one_class_greedy(
                var, vae, cid, args.per_class,
                teacher_pref.to(device), uni_bias.to(device),
                vnums, coarse_K=K, device=device
            )

            imgs = _decode_tokens(vae, ids, vnums)
            for n, img in enumerate(imgs):
                p = save_dir / f"img_{n:02d}.png"
                save_image(img, str(p))
                manifest["items"].append({
                    "scale_idx": i, "K": int(K),
                    "class_id": int(cid), "class_name": cname,
                    "path": str(p)
                })
                total_saved += 1

            all_idx_np.append(ids.detach().cpu().numpy())
            all_lab_np.append(np.full((ids.size(0),), cid, dtype=np.int64))

        if args.save_tokens:
            gen_idx = np.concatenate(all_idx_np, axis=0)
            gen_lab = np.concatenate(all_lab_np, axis=0)
            np.savez_compressed(
                comb_dir / "tokens_gen.npz",
                idx=gen_idx, label=gen_lab,
                vnums=np.asarray(vnums, dtype=np.int64)
            )

        print(f"[{i:02d}/{len(coarse_K_list)}] K={K} (GREEDY) → saved {args.per_class * num_classes} images at {comb_dir}")

    (sweep_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    expect = len(coarse_K_list) * num_classes * args.per_class
    print(f"✅ Done. Saved {total_saved} images under: {sweep_root}")
    if total_saved != expect:
        print(f"⚠️ Expect {expect}, got {total_saved}. Please check classes or args.")

if __name__ == "__main__":
    main()
