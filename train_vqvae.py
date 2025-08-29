"""
Multi-scale VQ-VAE trainer

Features:
- Tracks sum MSE / VQ / Total losses for train / val / test.
- Saves loss_matrix.json including codebook stats (active count / avg usage).
- Plots:
    - mse_curve_log.png        (train/val/test MSE, log Y)
    - mse_curve_train_val_log.png (train/val MSE, log Y)
    - train_components.png     (train MSE / VQ / Total)
- Reconstruction snapshots
- Codebook weights saved as both .npy and .pth
"""

from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow truncated images

import json, time, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.vqvae import VQVAE

# ----------------------- Paths & Hyperparams -----------------------
DATA_ROOT = "/home/polyaxon-data/data1/Tim&Yiping/Datasets_preprocessed_Yiping/Datasets_split1/train"
VAL_ROOT  = "/home/polyaxon-data/data1/Tim&Yiping/Datasets_preprocessed_Yiping/Datasets_split1/val"
TEST_ROOT = "/home/polyaxon-data/data1/Tim&Yiping/Datasets_preprocessed_Yiping/Datasets_split1/test"
OUT       = Path("/home/guests/yiping_zhou/projects/ultrasound_VAR/outputs/sum_mse_loss_with_usage2")

BATCH, EPOCHS = 4, 100
LR            = 2e-4
VOCAB_SIZE    = 2048
SNAP_EPOCHS   = [1, 30, 60, 100]
SEED          = 0
# ------------------------------------------------------------------

# CUDA-safe torch.bincount shim (some environments crash on CUDA tensors)
_orig_bc = torch.bincount
def _safe_bc(x, *a, **k):
    if x.dtype != torch.long:
        x = x.long()
    return (_orig_bc(x.cpu(), *a, **k) if x.is_cuda else _orig_bc(x, *a, **k)).to(x.device)
torch.bincount = _safe_bc

# ----------------------------- Data --------------------------------
from torchvision.datasets.folder import default_loader

def safe_loader(path: str):
    """Return None for corrupt images instead of raising."""
    try:
        return default_loader(path)
    except (OSError, UnidentifiedImageError) as e:
        print(f"⚠  Skip bad image: {path} | {e}")
        return None

class FilterBad(datasets.ImageFolder):
    """ImageFolder that skips images whose loader returns None."""
    def __init__(self, root, transform):
        super().__init__(root, transform=transform, loader=safe_loader)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        while sample is None:  # on failure, try the next sample
            index = (index + 1) % len(self.samples)
            sample, target = super().__getitem__(index)
        return sample, target

def build_loader(root: str, bs: int, shuffle: bool):
    # NOTE: Using the same transform for all splits to preserve original behavior.
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    ds = FilterBad(root, tfm)
    return DataLoader(
        ds, bs, shuffle=shuffle, num_workers=4, pin_memory=True,
        persistent_workers=False, drop_last=shuffle
    ), ds.classes
# -------------------------------------------------------------------

# ----------------------------- Plots --------------------------------
def plot_curve(data: dict, ylab: str, save: Path, logy: bool = False):
    x = np.arange(1, len(next(iter(data.values()))) + 1)
    plt.figure(); plt.grid(True)
    if logy: plt.yscale("log")
    for lbl, vals in data.items():
        plt.plot(x, vals, label=lbl)
    plt.xlabel("Epoch"); plt.ylabel(ylab); plt.legend()
    plt.tight_layout(); plt.savefig(save, dpi=200); plt.close()

def recon_pair(img: torch.Tensor, rec: torch.Tensor, save: Path, title: str):
    plt.figure(figsize=(4, 2)); plt.suptitle(title, fontsize=8)
    plt.subplot(1, 2, 1); plt.axis("off"); plt.imshow(img.permute(1, 2, 0))
    plt.subplot(1, 2, 2); plt.axis("off"); plt.imshow(rec.clamp(0, 1).permute(1, 2, 0))
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
# --------------------------------------------------------------------

def main():
    torch.manual_seed(SEED); random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output dirs
    (OUT / "recon").mkdir(parents=True, exist_ok=True)
    (OUT / "codebook").mkdir(exist_ok=True)

    # Data
    tr_loader, classes = build_loader(DATA_ROOT, BATCH, True)
    va_loader, _       = build_loader(VAL_ROOT,  BATCH, False)
    te_loader, _       = build_loader(TEST_ROOT, BATCH, False)

    # Model & optim
    model = VQVAE(
        VOCAB_SIZE, 16, 64, beta=0.25,
        test_mode=False,  # explicit: training mode
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=1e-5
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    # History & matrix
    hist = defaultdict(list)  # curves
    matrix = []               # JSON rows
    best_val = float("inf")

    # ---------------------- Evaluation ----------------------
    def evaluate(loader):
        model.eval(); mse = vq = n = 0
        with torch.no_grad():
            for img, _ in loader:
                img = img.to(device)
                rec, _, vq_l = model(img)
                mse += F.mse_loss(rec, img, reduction="sum").item()
                vq  += vq_l.item() * img.size(0)
                n   += img.size(0)
        mse /= n; vq /= n
        return mse, vq, mse + vq
    # --------------------------------------------------------

    # ------------------------- Train ------------------------
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        mse = vq = n = 0

        for img, _ in tr_loader:
            img = img.to(device)
            rec, _, vq_l = model(img)
            mse_l = F.mse_loss(rec, img, reduction="sum") / img.size(0)  # per-image sum-MSE
            loss  = mse_l + vq_l

            opt.zero_grad()
            loss.backward()
            opt.step()

            mse += mse_l.item() * img.size(0)
            vq  += vq_l.item()  * img.size(0)
            n   += img.size(0)

        sched.step()

        # Averages
        tr_mse, tr_vq = mse / n, vq / n
        tr_tot = tr_mse + tr_vq
        va_mse, va_vq, va_tot = evaluate(va_loader)
        te_mse, te_vq, te_tot = evaluate(te_loader)

        # Codebook stats
        usage = model.quantize.ema_vocab_hit_SV.sum(0).cpu().numpy()
        matrix.append({
            "epoch": ep,
            "train": {"mse": tr_mse, "vq": tr_vq, "total": tr_tot},
            "val":   {"mse": va_mse, "vq": va_vq, "total": va_tot},
            "test":  {"mse": te_mse, "vq": te_vq, "total": te_tot},
            "codebook": {
                "active": int((usage > 0).sum()),
                "avg_usage": float(usage.mean())
            }
        })

        # Curves
        for tag, val in [("train", tr_mse), ("val", va_mse), ("test", te_mse)]:
            hist[f"{tag}_mse"].append(val)
        hist["tr_vq"].append(tr_vq)
        hist["tr_tot"].append(tr_tot)

        print(
            f"[{ep:03d}/{EPOCHS}] "
            f"tr_tot {tr_tot:.4e}  va_tot {va_tot:.4e}  "
            f"active {int((usage > 0).sum())}/{VOCAB_SIZE}  "
            f"({time.time() - t0:.1f}s)"
        )

        # Recon snapshots
        if ep in SNAP_EPOCHS:
            model.eval()
            for split, loader in [("train", tr_loader), ("val", va_loader), ("test", te_loader)]:
                seen = {c: False for c in classes}  # first sample per class
                with torch.no_grad():
                    for img, label in loader:
                        for i, cid in enumerate(label):
                            cls = classes[cid]
                            if not seen[cls]:
                                inp = img[i:i+1].to(device)
                                rec, _, _ = model(inp)
                                recon_pair(
                                    inp[0].cpu(), rec[0].cpu(),
                                    OUT / f"recon/{split}-{cls}-ep{ep}.png",
                                    f"{split}-{cls}-ep{ep}"
                                )
                                seen[cls] = True
                        if all(seen.values()):
                            break

        # Save best model (by val total)
        if va_tot < best_val:
            best_val = va_tot
            torch.save(model.state_dict(), OUT / "best.pth")
    # --------------------------------------------------------

    # ------------------------ Finalize ----------------------
    with open(OUT / "loss_matrix.json", "w") as f:
        json.dump(matrix, f, indent=2)

    plot_curve(
        {"train": hist["train_mse"], "val": hist["val_mse"], "test": hist["test_mse"]},
        "MSE / pixel",
        OUT / "mse_curve_log.png",
        logy=True
    )

    plot_curve(
        {"train": hist["train_mse"], "val": hist["val_mse"]},
        "Train & Val MSE / pixel",
        OUT / "mse_curve_train_val_log.png",
        logy=True
    )

    plot_curve(
        {"mse": hist["train_mse"], "vq": hist["tr_vq"], "total": hist["tr_tot"]},
        "train losses",
        OUT / "train_components.png"
    )

    # Codebook weights
    cb = model.quantize.embedding.weight.detach().cpu()
    np.save(OUT / "codebook/codebook_weight.npy", cb.numpy())
    torch.save(cb, OUT / "codebook/codebook_weight.pth")

    torch.save(model.state_dict(), OUT / "last_model.pth")
    print("✅ All done ->", OUT)
    # --------------------------------------------------------

if __name__ == "__main__":
    main()
