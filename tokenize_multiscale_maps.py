"""
Tokenize a dataset into multi-scale VQ-VAE code maps.

Outputs (under --out):
- tokens_multiscale_maps.npz: { tok_{v}x{v}: (N,v,v), vnums, idx (N,L), label (N,) }
- tokens.npz:                { idx (N,L), label (N,) }   # compatibility
- classes.json:              ImageFolder class list
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms, datasets

from models.vqvae import VQVAE


def get_vnums_from_vae(vae, fallback=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)):
    """Return VQ-VAE's multi-scale patch sizes (v) as a tuple."""
    if hasattr(vae, "quantize") and hasattr(vae.quantize, "v_patch_nums"):
        return tuple(int(x) for x in vae.quantize.v_patch_nums)
    if hasattr(vae, "v_patch_nums"):
        return tuple(int(x) for x in vae.v_patch_nums)
    return tuple(fallback)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- VQ-VAE ---
    vae = VQVAE(
        vocab_size=args.vocab, z_channels=16, ch=64, beta=0.25,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    ).to(device)
    vae.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    vae.eval()

    vnums = get_vnums_from_vae(vae)

    # --- Transforms (match training preprocessing) ---
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(args.root, tfm)

    # --- Output dir ---
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Buffers
    scale_buffers = {v: [] for v in vnums}  # v -> list of (v,v)
    idx_concat = []                         # (N, L) concatenated tokens
    labels = []

    with torch.no_grad():
        pbar = tqdm(ds, desc="tokenizing (per-scale 2D + concat)", ncols=100)
        for img, lab in pbar:
            img = img.unsqueeze(0).to(device)  # (1,C,H,W)

            # Multi-scale encode: List[Tensor(1, v*v)] aligned with vnums
            ms_idx = vae.img_to_idxBl(img)
            assert len(ms_idx) == len(vnums), "Mismatch in returned scales vs v_patch_nums"

            concat_this = []
            for blk, v in zip(ms_idx, vnums):
                flat = blk.view(-1)  # (v*v,)
                grid = flat.reshape(v, v).detach().cpu().numpy().astype(np.int64)  # (v,v)
                scale_buffers[v].append(grid)
                concat_this.append(flat.detach().cpu().numpy().astype(np.int64))

            idx_concat.append(np.concatenate(concat_this, axis=0))  # (L,)
            labels.append(np.int64(lab))

    # --- Assemble & save ---
    if len(idx_concat) == 0:
        raise RuntimeError("No samples processed. Check --root path.")

    idx_arr = np.stack(idx_concat, axis=0).astype(np.int64)  # (N, L)
    label_arr = np.asarray(labels, dtype=np.int64)

    # Basic checks
    if (idx_arr < 0).any():
        raise ValueError("Found negative token id.")
    vmax = int(idx_arr.max())
    if vmax >= args.vocab:
        raise ValueError(f"Token id {vmax} >= vocab_size {args.vocab}. Check codebook/encoder.")

    # Per-scale arrays
    per_scale_np = {}
    for v in vnums:
        arr = np.stack(scale_buffers[v], axis=0).astype(np.int64)  # (N, v, v)
        per_scale_np[f"tok_{v}x{v}"] = arr

    per_scale_np["vnums"] = np.asarray(vnums, dtype=np.int32)
    per_scale_np["idx"] = idx_arr
    per_scale_np["label"] = label_arr

    # Main NPZ
    np.savez_compressed(out_dir / "tokens_multiscale_maps.npz", **per_scale_np)

    # Compatibility NPZ (optional)
    np.savez_compressed(out_dir / "tokens.npz", idx=idx_arr, label=label_arr)

    # Classes
    with open(out_dir / "classes.json", "w") as f:
        json.dump(ds.classes, f)

    # Log
    print(f"Done: {out_dir}")
    print(f"  idx (concat) shape: {idx_arr.shape}, dtype: {idx_arr.dtype}")
    for v in vnums:
        arr = per_scale_np[f"tok_{v}x{v}"]
        print(f"  tok_{v}x{v}: {arr.shape}, dtype: {arr.dtype}")
    print(f"  token range: [{int(idx_arr.min())}, {int(idx_arr.max())}]")
    print(f"  vnums: {vnums}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize an ImageFolder into multi-scale VQ-VAE code maps."
    )
    parser.add_argument("--root", required=True, help="Dataset root split (e.g., .../train)")
    parser.add_argument("--ckpt", required=True, help="VQ-VAE checkpoint (best.pth)")
    parser.add_argument("--out",  required=True, help="Output folder (e.g., data_tokens/train)")
    parser.add_argument("--vocab", type=int, default=2048, help="Codebook size")
    main(parser.parse_args())
