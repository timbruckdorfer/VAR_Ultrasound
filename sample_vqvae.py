#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run-only inference to compute per-class reconstruction metrics for
{train, val, test} × classes. Outputs a single JSON with 36 entries like:

"/.../train/Liver_Normal": {
  "mse": 0.00118, "std_mse": 0.00049,
  "mae": 0.01827, "std_mae": 0.00383,
  "psnr": 29.65,   "std_psnr": 1.8183,
  "num_images": 30
}
"""

from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse, json, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# ---------- SAFE PATCH: avoid dist error triggered in quant.py ----------
from torch import distributed as tdist
if not hasattr(tdist, "_orig_get_world_size"):
    tdist._orig_get_world_size = tdist.get_world_size
def _safe_get_world_size(*args, **kwargs):
    try:
        if hasattr(tdist, "is_initialized") and tdist.is_initialized():
            return tdist._orig_get_world_size(*args, **kwargs)
    except Exception:
        pass
    return 1
tdist.get_world_size = _safe_get_world_size
# -----------------------------------------------------------------------

from models.vqvae import VQVAE


# ---- Safe loader to skip corrupt files ----
from torchvision.datasets.folder import default_loader
def safe_loader(path: str):
    try:
        return default_loader(path)
    except (OSError, UnidentifiedImageError) as e:
        print(f"⚠  Skip bad image: {path} | {e}")
        return None

class FilterBad(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform, loader=safe_loader)
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        while sample is None:
            index = (index + 1) % len(self.samples)
            sample, target = super().__getitem__(index)
        return sample, target
# ------------------------------------------


@torch.no_grad()
def eval_split(root: str, vae: VQVAE, device: str, batch: int, workers: int):
    """Return dict: {full_class_path: metrics} for this split."""
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    ds = FilterBad(root, tfm)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False
    )
    classes = ds.classes
    # per-class accumulators
    acc = {
        cls: {"mse": [], "mae": [], "psnr": []}
        for cls in classes
    }

    for imgs, labels in dl:
        imgs = imgs.to(device, non_blocking=True)
        # -------- encode -> tokens -> decode  (NO vae.forward()) --------
        idx_Bl = vae.img_to_idxBl(imgs)  # list of [B, l] per scale
        recs   = vae.idxBl_to_img(idx_Bl, same_shape=True, last_one=True)  # [B,3,H,W]
        # ----------------------------------------------------------------
        # per-image metrics (per-pixel)
        per_mse  = F.mse_loss(recs, imgs, reduction="none").view(imgs.size(0), -1).mean(dim=1)  # [B]
        per_mae  = F.l1_loss (recs, imgs, reduction="none").view(imgs.size(0), -1).mean(dim=1)  # [B]
        per_psnr = -10.0 * torch.log10(torch.clamp(per_mse, min=1e-10))                         # [B]

        for i, lab in enumerate(labels):
            cls = classes[int(lab)]
            acc[cls]["mse"].append(float(per_mse[i].item()))
            acc[cls]["mae"].append(float(per_mae[i].item()))
            acc[cls]["psnr"].append(float(per_psnr[i].item()))

    # reduce to mean/std and attach absolute path keys
    out = {}
    for cls in classes:
        mse  = np.array(acc[cls]["mse"], dtype=np.float64)
        mae  = np.array(acc[cls]["mae"], dtype=np.float64)
        psnr = np.array(acc[cls]["psnr"], dtype=np.float64)
        n = int(mse.size)

        key = str(Path(root) / cls)  # e.g., .../train/Liver_Normal
        if n == 0:
            out[key] = {"mse": None, "mae": None, "psnr": None,
                        "std_mse": None, "std_mae": None, "std_psnr": None,
                        "num_images": 0}
        else:
            out[key] = {
                "mse": float(mse.mean()), "std_mse": float(mse.std(ddof=0)),
                "mae": float(mae.mean()), "std_mae": float(mae.std(ddof=0)),
                "psnr": float(psnr.mean()), "std_psnr": float(psnr.std(ddof=0)),
                "num_images": n
            }
    return out


def load_vae(ckpt_dir: str, ckpt_tag: str, codebook_path: str|None, device: str) -> VQVAE:
    vae = VQVAE(
        vocab_size=2048, z_channels=16, ch=64, beta=0.25,
        test_mode=True,
        v_patch_nums=(1,2,3,4,5,6,8,10,13,16)
    ).to(device)

    ckpt_dir = Path(ckpt_dir)
    if ckpt_tag == "best":
        ckpt = ckpt_dir / "best.pth"
        if not ckpt.exists():
            print("⚠ best.pth not found, fallback to last_model.pth")
            ckpt = ckpt_dir / "last_model.pth"
    elif ckpt_tag == "last":
        ckpt = ckpt_dir / "last_model.pth"
    else:
        ckpt = Path(ckpt_tag)
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    vae.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

    if codebook_path:
        cb = torch.load(codebook_path, map_location="cpu")
        with torch.no_grad():
            vae.quantize.embedding.weight.data.copy_(cb)

    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    print(f"✅ VAE loaded: {ckpt}")
    return vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root",   required=True)
    ap.add_argument("--test_root",  required=True)
    ap.add_argument("--ckpt_dir",   required=True, help="folder that contains best.pth / last_model.pth")
    ap.add_argument("--ckpt", choices=["best","last","path"], default="best",
                    help="'best'|'last' or an explicit path when --ckpt=path")
    ap.add_argument("--ckpt_path", type=str, default="", help="used when --ckpt=path")
    ap.add_argument("--codebook", type=str, default="", help="optional codebook .pth")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_json", type=str, default="recon_matrix_all.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_tag = args.ckpt_path if args.ckpt == "path" else args.ckpt
    codebook = args.codebook if args.codebook else None
    vae = load_vae(args.ckpt_dir, ckpt_tag, codebook, device=device)

    result = {}
    for root in [args.train_root, args.val_root, args.test_root]:
        print(f"\n[->] Evaluating split: {root}")
        part = eval_split(root, vae, device, batch=args.batch, workers=args.workers)
        result.update(part)

    save_to = Path(args.ckpt_dir) / args.save_json
    with open(save_to, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved matrix to: {save_to}")
    print(f"Total entries: {len(result)} (should be 36 if 12 classes × 3 splits)")

if __name__ == "__main__":
    main()
