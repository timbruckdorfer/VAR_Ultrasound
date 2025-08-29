"""
Metrics for VAR (Next-Scale) samples.

Inputs (CLI):
- --out_dir: directory that contains generated sweep results.
- --tok_dir: directory that contains the real train tokens (tokens_multiscale_maps.npz) and classes.json.
- --vae_ckpt: VQ-VAE checkpoint (.pth), used to obtain codebook embeddings.
- --codebook: codebook weights (.pth).

Computes:
  1) KID (on codebook-embedding features), overall and per-scale.
  2) Codebook usage ratio (unique token ratio) and entropy, overall and per-scale.
  3) Diversity of generated set: mean pairwise cosine on overall features (lower is more diverse).

Outputs:
- Two CSV files under the sweep folder:
  - metrics_overall.csv
  - metrics_per_scale.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch

from models.vqvae import VQVAE


# ---------- utils ----------
def load_real_tokens(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    d = np.load(npz_path, allow_pickle=False)
    vnums = tuple(int(x) for x in d["vnums"].tolist()) if "vnums" in d else (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    lens = [v * v for v in vnums]
    Ltot = int(sum(lens))
    labels = d["label"].astype(np.int64)

    N = d[f"tok_{vnums[0]}x{vnums[0]}"].shape[0]
    idx1d = np.empty((N, Ltot), dtype=np.int64)
    for i in range(N):
        off = 0
        for v in vnums:
            tok = d[f"tok_{v}x{v}"][i].reshape(-1)
            idx1d[i, off:off + v * v] = tok
            off += v * v
    return idx1d, labels, vnums


def seg_bounds(vnums: Tuple[int, ...]) -> np.ndarray:
    lens = [v * v for v in vnums]
    return np.cumsum([0] + lens)


def codebook_embed(embedding: torch.Tensor, idx_np: np.ndarray) -> torch.Tensor:
    """Pool token embeddings overall -> [N, D]."""
    idx = torch.from_numpy(idx_np).long().to(embedding.device)
    feat = embedding.index_select(0, idx.view(-1)).view(idx.size(0), idx.size(1), -1)  # [N, L, D]
    return feat.mean(dim=1)  # [N, D]


def codebook_embed_per_scale(embedding: torch.Tensor, idx_np: np.ndarray, vnums: Tuple[int, ...]) -> List[torch.Tensor]:
    """Pool token embeddings per scale -> list of [N, D]."""
    idx = torch.from_numpy(idx_np).long().to(embedding.device)
    lens = [v * v for v in vnums]
    seg = seg_bounds(vnums)
    emb = embedding.index_select(0, idx.view(-1)).view(idx.size(0), idx.size(1), -1)  # [N, L, D]
    out = []
    for si in range(len(lens)):
        st, ed = int(seg[si]), int(seg[si + 1])
        out.append(emb[:, st:ed, :].mean(dim=1))  # [N, D]
    return out


# --- KID (polynomial kernel, unbiased MMD^2) ---
def _poly_kernel(x: torch.Tensor, y: torch.Tensor, degree=3) -> torch.Tensor:
    # Typical KID kernel: ((x·y / d) + 1)^degree
    d = x.size(-1)
    return (x @ y.t() / float(d) + 1.0) ** degree


def _mmd2_unbiased(Kxx, Kyy, Kxy) -> torch.Tensor:
    n = Kxx.size(0)
    m = Kyy.size(0)
    Kxx_sum = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1) + 1e-8)
    Kyy_sum = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1) + 1e-8)
    Kxy_mean = Kxy.mean()
    return Kxx_sum + Kyy_sum - 2.0 * Kxy_mean


@torch.no_grad()
def kid_mmd2(x: torch.Tensor, y: torch.Tensor, degree=3) -> float:
    Kxx = _poly_kernel(x, x, degree)
    Kyy = _poly_kernel(y, y, degree)
    Kxy = _poly_kernel(x, y, degree)
    return float(_mmd2_unbiased(Kxx, Kyy, Kxy).cpu().item())


# --- usage & entropy ---
def usage_and_entropy(idx_np: np.ndarray, vocab: int, st: int = None, ed: int = None) -> Tuple[float, float]:
    if st is None:
        st = 0
    if ed is None:
        ed = idx_np.shape[1]
    seg = idx_np[:, st:ed].reshape(-1)
    counts = np.bincount(seg, minlength=vocab).astype(np.float64)
    used = (counts > 0).sum()
    p = counts / max(1.0, counts.sum())
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())  # nats
    return float(used / float(vocab)), entropy


# --- diversity: mean pairwise cosine on overall features ---
def mean_pairwise_cosine(feat: torch.Tensor) -> float:
    feat = torch.nn.functional.normalize(feat, dim=1)
    sim = feat @ feat.t()  # [N, N]
    n = sim.size(0)
    return float((sim.sum() - sim.diag().sum()).cpu().item() / (n * (n - 1) + 1e-8))


def main():
    ap = argparse.ArgumentParser(description="Compute metrics for VAR (Next-Scale) samples.")
    ap.add_argument("--out_dir", required=True, help="Output directory that contains sweep results (samples_sweep3).")
    ap.add_argument("--tok_dir", required=True, help="Directory that provides train/tokens_multiscale_maps.npz and classes.json.")
    ap.add_argument("--vae_ckpt", required=True, help="VQ-VAE checkpoint (.pth), used to fetch the codebook embedding.")
    ap.add_argument("--codebook", required=True, help="Codebook weights (.pth).")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--degree", type=int, default=3, help="Polynomial kernel degree for KID (default: 3).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    sweep_root = out_dir / "samples_sweep3"
    assert sweep_root.exists(), f"Missing sweep folder: {sweep_root}. Run sampling first."

    tok_train = Path(args.tok_dir) / "train" / "tokens_multiscale_maps.npz"
    idx_real, lab_real, vnums = load_real_tokens(tok_train)
    seg = seg_bounds(vnums)
    vocab = args.vocab

    # VQVAE only to get the codebook embedding
    vae = VQVAE(vocab_size=vocab, z_channels=16, ch=64, beta=0.25, test_mode=True, v_patch_nums=vnums).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"), strict=False)
    with torch.no_grad():
        vae.quantize.embedding.weight.data.copy_(torch.load(args.codebook, map_location="cpu"))
    vae.eval()
    emb = vae.quantize.embedding.weight.detach()  # [V, D]

    # Real-set features
    feat_real_overall = codebook_embed(emb, idx_real)
    feat_real_scales = codebook_embed_per_scale(emb, idx_real, vnums)

    # CSV buffers
    overall_rows = [["scale_idx", "K", "tag", "kid_overall", "diversity_cosine", "usage_ratio_overall", "entropy_overall"]]
    scale_rows = [["scale_idx", "K", "tag", "kid_scale", "usage_ratio_scale", "entropy_scale"]]

    # Iterate combinations
    for scale_dir in sorted(sweep_root.glob("scale_*")):
        name = scale_dir.name  # e.g., scale_01_K0001
        try:
            parts = name.split("_")
            scale_idx = int(parts[1])
            K = int(parts[2][1:])  # "K0001" -> 1
        except Exception:
            print(f"Skip directory (cannot parse name): {name}")
            continue

        for tag_dir in ["TP", "NOP"]:
            comb = scale_dir / tag_dir
            if not comb.exists():
                continue
            npz_path = comb / "tokens_gen.npz"
            if not npz_path.exists():
                print(f"Missing file: {npz_path} (skipped)")
                continue

            d = np.load(npz_path, allow_pickle=False)
            idx_gen = d["idx"]

            # overall
            feat_gen_overall = codebook_embed(emb, idx_gen)
            kid_overall = kid_mmd2(feat_gen_overall, feat_real_overall, degree=args.degree)
            diversity = mean_pairwise_cosine(feat_gen_overall)
            usage_overall, ent_overall = usage_and_entropy(idx_gen, vocab)
            overall_rows.append([scale_idx, K, tag_dir, kid_overall, diversity, usage_overall, ent_overall])

            # per-scale
            feat_gen_scales = codebook_embed_per_scale(emb, idx_gen, vnums)
            for si, (fr, fg) in enumerate(zip(feat_real_scales, feat_gen_scales), start=1):
                kid_s = kid_mmd2(fg, fr, degree=args.degree)
                st, ed = int(seg[si - 1]), int(seg[si])
                usage_s, ent_s = usage_and_entropy(idx_gen, vocab, st, ed)
                scale_rows.append([scale_idx, K, tag_dir, kid_s, usage_s, ent_s])

    # Save CSVs
    overall_csv = sweep_root / "metrics_overall.csv"
    scale_csv = sweep_root / "metrics_per_scale.csv"
    with open(overall_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(overall_rows)
    with open(scale_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(scale_rows)

    print("Metrics saved:")
    print(" -", overall_csv)
    print(" -", scale_csv)


if __name__ == "__main__":
    main()
