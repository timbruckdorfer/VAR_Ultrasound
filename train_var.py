#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VAR trainer (online tokens) + robust 'sampling_pack' exporter

Exports after training:
  OUT_DIR/sampling_pack/
    - classes.json
    - vnums.json
    - tokens_multiscale_maps.npz
    - stats_full.npz      (pos_prior/teacher_pref/uni_bias/vnums/lens)
    - teacher_bank.npz    (always saved if TP_BANK_MAX >= 0; 0 means 'no limit')
    - meta.json           (L_total/vocab/C/K_edges)
"""

import os, time, random, csv, gc, json, math
from pathlib import Path
from contextlib import nullcontext
from typing import List, Tuple, Optional, Dict

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

# Limit CPU threads (avoid BLAS/DataLoader contention)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader as _default_loader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import faulthandler; faulthandler.enable()

# ====== paths ======
DATA_ROOT_TRAIN = "/home/polyaxon-data/data1/Tim&Yiping/Datasets_preprocessed_Yiping/Datasets_split1/train"
DATA_ROOT_VAL   = "/home/polyaxon-data/data1/Tim&Yiping/Datasets_preprocessed_Yiping/Datasets_split1/val"

VAE_CKPT      = "/home/guests/yiping_zhou/projects/ultrasound_VAR/outputs/sum_mse_loss_with_usage2/best.pth"
CODEBOOK_PTH  = "/home/guests/yiping_zhou/projects/ultrasound_VAR/outputs/sum_mse_loss_with_usage2/codebook/codebook_weight.pth"

OUT_DIR = Path("/home/guests/yiping_zhou/projects/ultrasound_VAR/outputs/VAR_online_tokens_fix21")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(METRICS_DIR := OUT_DIR / "metrics").mkdir(exist_ok=True)
(SAMPLES_DIR := OUT_DIR / "samples").mkdir(exist_ok=True)

# ====== hparams ======
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_RESO    = 256
BATCH         = 4
NUM_WORKERS   = 2
EPOCHS        = 300
LR            = 1e-4
WEIGHT_DECAY  = 1e-2
VOCAB_SIZE    = 2048

DEPTH = 6
DIM   = 384
HEADS = 6

LABEL_SMOOTH = 0.05
USE_AMP      = True
AMP_BF16     = False
ACCUM_STEPS  = 4

# LR schedule
MIN_LR        = 3e-5
WARMUP_EPOCHS = 5
USE_COSINE    = True

# Progressive (optional)
USE_PROGRESSIVE = False
PG_START_STAGE  = 0
PG_PORTION      = 0.7
PG_WP_PORTION   = 0.05

# Eval & logging
EVAL_PERIOD          = 1
TRAIN_EVAL_PERIOD    = 5
SAVE_EVERY           = 1

# Sampling (quicklook during training)
DO_SAMPLING          = True
SAMPLE_EARLY_EPOCHS  = {1,2,3,4,5,10,20}
SAMPLE_EVERY         = 25
PER_CLASS_SAMPLE     = 1
COARSE_K             = 155
USE_TEACHER_PREFIX   = True
USE_COARSE_ARGMAX    = True
TEMP_MIN, TEMP_MAX   = 0.95, 0.80
TOPK_MIN, TOPK_MAX   = 64,  128
TOPP_MIN, TOPP_MAX   = 0.90, 0.95
REPEAT_PENALTY       = 1.20
REPEAT_WINDOW        = 64
UNIGRAM_ALPHA        = 0.3

# Export sampling pack
SAVE_SAMPLING_PACK = True         # enable exporter
TP_BANK_MAX        = 0            # 0 == keep ALL; >0 == per-class cap

# 变换（统计/采样统一无增强）
NOAUG_TF = transforms.Compose([
    transforms.Resize(INPUT_RESO),
    transforms.CenterCrop(INPUT_RESO),
    transforms.ToTensor(),
])

# ====== imports from your project ======
from models.vqvae import VQVAE
from models.var   import VAR

# ====== spawn start method ======
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ====== Data ======
train_tf = transforms.Compose([
    transforms.Resize(INPUT_RESO),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.CenterCrop(INPUT_RESO),
    transforms.ToTensor(),
])
val_tf = NOAUG_TF

class SafeImageFolder(datasets.ImageFolder):
    """Return None for a bad sample to let collate filter it out."""
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            return None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def build_loader(root: str, tfm, bs: int, shuffle: bool):
    ds = SafeImageFolder(root=root, transform=tfm)
    kwargs = dict(
        batch_size=bs,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=safe_collate,
        worker_init_fn=seed_worker,
        timeout=60,
    )
    if NUM_WORKERS > 0:
        kwargs["multiprocessing_context"] = mp.get_context("spawn")
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    dl = DataLoader(ds, **kwargs)
    return dl, ds

# ====== Utils ======
@torch.no_grad()
def _infer_total_stride(vae_model: VQVAE, input_reso: int) -> int:
    vae_model.eval()
    dev = next(vae_model.parameters()).device
    x = torch.zeros(1, 3, input_reso, input_reso, device=dev)
    z = vae_model.encoder(x)
    if hasattr(vae_model, "pre_vq_conv") and vae_model.pre_vq_conv is not None:
        z = vae_model.pre_vq_conv(z)
    h, w = z.shape[-2], z.shape[-1]
    assert input_reso % h == 0 and input_reso % w == 0
    sh, sw = input_reso // h, input_reso // w
    assert sh == sw
    return sh

def _split_lens_from_vnums(vnums: Tuple[int, ...]) -> Tuple[List[int], int, List[Tuple[int,int]]]:
    lens = [v*v for v in vnums]
    Ltot = sum(lens)
    seg  = np.cumsum([0] + lens)
    spans = [(int(seg[i]), int(seg[i+1])) for i in range(len(lens))]
    return lens, Ltot, spans

@torch.no_grad()
def build_var_inputs_from_imgs(vae: VQVAE, imgs: torch.Tensor):
    gt_idx_Bl = vae.img_to_idxBl(imgs)             # list of [B,l]
    gt_BL     = torch.cat(gt_idx_Bl, dim=1)        # [B, L]
    x_BLCv    = vae.quantize.idxBl_to_var_input(gt_idx_Bl)  # [B, L, C]
    return x_BLCv, gt_BL

# ====== Loss ======
ce_train_nored = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH, reduction='none')
ce_eval_mean   = nn.CrossEntropyLoss(label_smoothing=0.0,           reduction='mean')

def amp_ctx():
    if not USE_AMP or (DEVICE != "cuda"):
        return nullcontext()
    try:
        dtype = torch.bfloat16 if AMP_BF16 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported() else torch.float16
        return torch.cuda.amp.autocast(device_type="cuda", dtype=dtype)
    except TypeError:
        return torch.cuda.amp.autocast()

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and (DEVICE == "cuda"))

# ====== Train / Eval ======
def run_epoch_train(model: VAR, vae: VQVAE, loader: DataLoader, lens: List[int], L_TOTAL: int,
                    spans: List[Tuple[int,int]], epoch: int, total_epochs: int, optimizer: torch.optim.Optimizer):
    model.train(True)
    optimizer.zero_grad(set_to_none=True)

    use_pg = bool(USE_PROGRESSIVE)

    # uniform + tail emphasis
    LOSS_WEIGHT = torch.ones(1, L_TOTAL, device=DEVICE)
    st_tail = L_TOTAL - lens[-1]
    TAIL_WEIGHT = 2.0
    LOSS_WEIGHT[:, st_tail:] *= TAIL_WEIGHT
    LOSS_WEIGHT = LOSS_WEIGHT / LOSS_WEIGHT.sum()

    tot = 0; Lm_sum = Lt_sum = Am_sum = At_sum = 0.0
    V = VOCAB_SIZE

    for it, batch in enumerate(loader):
        if batch is None:
            continue
        imgs, labels = batch

        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, dtype=torch.long, non_blocking=True)
        B = imgs.size(0)

        x_BLCv, gt_BL = build_var_inputs_from_imgs(vae, imgs)

        with amp_ctx():
            logits_BLV = model(labels, x_BLCv)  # [B, L, V]
            loss_BL = ce_train_nored(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)

            if use_pg:
                max_it = total_epochs * len(loader)
                g_it = (epoch-1)*len(loader) + it
                wp_it  = PG_WP_PORTION * max_it
                pg_end = PG_PORTION * max_it
                if g_it <= wp_it: prog_si = PG_START_STAGE
                elif g_it >= pg_end: prog_si = len(lens) - 1
                else:
                    delta = (len(lens) - 1 - PG_START_STAGE)
                    progress = min(max((g_it - wp_it) / max(1, (pg_end - wp_it)), 0.0), 1.0)
                    prog_si = PG_START_STAGE + round(progress * delta)
                bg, ed = spans[prog_si]
                lw = LOSS_WEIGHT.clone()
                prog_wp = max(min((it+1) / max(1, PG_WP_PORTION*len(loader)), 1.0), 0.01)
                lw[:, bg:ed] = lw[:, bg:ed] * float(prog_wp)
            else:
                lw = LOSS_WEIGHT

            loss = loss_BL.mul(lw).sum(dim=-1).mean()

        do_step = (((it+1) % max(1, ACCUM_STEPS)) == 0) or (it+1 == len(loader))
        if scaler.is_enabled():
            scaler.scale(loss / max(1, ACCUM_STEPS)).backward()
            if do_step:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            (loss / max(1, ACCUM_STEPS)).backward()
            if do_step:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            Lm = ce_eval_mean(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).item()
            pred_BL = logits_BLV.argmax(dim=-1)
            Am = (pred_BL == gt_BL).sum().item() * (100.0 / gt_BL.shape[1])

            st_tail = L_TOTAL - lens[-1]
            Lt = ce_eval_mean(
                logits_BLV[:, st_tail:].reshape(-1, V),
                gt_BL[:,    st_tail:].reshape(-1)
            ).item()
            At = (pred_BL[:, st_tail:] == gt_BL[:, st_tail:]).sum().item() * (100.0 / lens[-1])

            Lm_sum += Lm * B; Lt_sum += Lt * B; Am_sum += Am; At_sum += At; tot += B

    return (Lm_sum/max(1,tot), Lt_sum/max(1,tot), Am_sum/max(1,tot), At_sum/max(1,tot))

@torch.no_grad()
def evaluate(model: VAR, vae: VQVAE, loader: DataLoader, lens: List[int], L_TOTAL: int):
    model.eval()
    tot = 0
    L_mean = L_tail = acc_mean = acc_tail = 0.0
    V = VOCAB_SIZE

    for batch in loader:
        if batch is None:
            continue
        imgs, labels = batch

        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, dtype=torch.long, non_blocking=True)
        B = imgs.size(0)

        x_BLCv, gt_BL = build_var_inputs_from_imgs(vae, imgs)
        logits_BLV = model(labels, x_BLCv)

        L_mean += ce_eval_mean(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).item() * B
        st_tail = L_TOTAL - lens[-1]
        L_tail += ce_eval_mean(
            logits_BLV[:, st_tail:].reshape(-1, V),
            gt_BL[:,    st_tail:].reshape(-1)
        ).item() * B

        pred_BL = logits_BLV.argmax(dim=-1)
        acc_mean += (pred_BL == gt_BL).sum().item() * (100.0 / gt_BL.shape[1])
        acc_tail += (pred_BL[:, st_tail:] == gt_BL[:, st_tail:]).sum().item() * (100.0 / lens[-1])
        tot += B

    tot = max(1, tot)
    return (L_mean / tot, L_tail / tot, acc_mean / tot, acc_tail / tot)

# ====== Sampling during training (optional) ======
@torch.no_grad()
def build_sampling_stats_online(vae: VQVAE, train_root: str, classes: List[str], K: int,
                                max_per_class: Optional[int] = None):
    base_ds = datasets.ImageFolder(root=train_root, transform=NOAUG_TF)
    C = len(classes); V = VOCAB_SIZE

    vnums = tuple(int(x) for x in vae.quantize.v_patch_nums)
    lens = [v*v for v in vnums]
    Ltot = sum(lens)
    K = min(K, Ltot)

    idxs_per_class: Dict[int, List[int]] = {c: [] for c in range(C)}
    for i, (_, tgt) in enumerate(base_ds.samples):
        idxs_per_class[tgt].append(i)

    counts = np.full((C, K, V), 0.5, dtype=np.float64)
    teacher = -np.ones((C, K), dtype=np.int64)
    uni_cnt = np.zeros((V,), dtype=np.int64)

    for c in range(C):
        idx_list = idxs_per_class[c]
        if len(idx_list) == 0:
            teacher[c, :] = 0
            continue
        use_idx = idx_list if (max_per_class is None) else idx_list[:max_per_class]

        for j, i in enumerate(use_idx):
            try:
                img, _ = base_ds[i]
            except Exception:
                continue
            img = img.unsqueeze(0).to(DEVICE)
            _, gt = build_var_inputs_from_imgs(vae, img)
            tok = gt[0, :K].cpu().numpy()

            if j == 0:
                teacher[c, :] = tok.copy()
            counts[c, np.arange(K), tok] += 1

            flat = gt.view(-1).cpu().numpy()
            uni_cnt += np.bincount(flat, minlength=V)

    counts /= counts.sum(axis=-1, keepdims=True)
    pos_prior = torch.from_numpy(counts).float().to(DEVICE)
    teacher   = torch.from_numpy(np.where(teacher<0, 0, teacher)).long().to(DEVICE)

    uni_prob  = uni_cnt / max(1, int(uni_cnt.sum()))
    uni_bias  = torch.from_numpy(UNIGRAM_ALPHA * np.log(uni_prob + 1e-8)).float().to(DEVICE)
    return pos_prior, teacher, uni_bias, lens, Ltot, vnums

@torch.no_grad()
def maybe_sample_and_save(ep: int, vae: VQVAE, model: VAR, train_root: str, class_names: List[str],
                          out_dir: Path, num_per_class: int = 1, coarse_K: int = COARSE_K,
                          use_teacher: bool = USE_TEACHER_PREFIX):
    pos_prior, teacher_pref, uni_bias, lens, L_total, vnums = build_sampling_stats_online(
        vae, train_root, class_names, coarse_K, max_per_class=None
    )
    S = len(lens)
    seg = np.cumsum([0] + lens)
    def span(si: int): return int(seg[si]), int(seg[si+1])
    def lin(a,b,s):   return a + (b-a) * (s / max(1, S-1))

    m = model
    prev = m.training; m.eval()
    to_pil = transforms.ToPILImage()

    C = len(class_names)
    for cid in range(C):
        B = num_per_class
        lab = torch.full((B,), cid, device=DEVICE, dtype=torch.long)
        y_tokens = torch.full((B, L_total), -1, device=DEVICE, dtype=torch.long)

        K = min(coarse_K, L_total)
        if use_teacher:
            y_tokens[:, :K] = teacher_pref[cid].unsqueeze(0).expand(B, -1)[:, :K]
        else:
            for p in range(K):
                distv = pos_prior[cid, p].unsqueeze(0).expand(B, -1)
                tok   = distv.argmax(-1, keepdim=True) if USE_COARSE_ARGMAX else torch.multinomial(distv, 1)
                y_tokens[:, p] = tok.squeeze(1)

        for s_idx in range(S):
            st, ed = span(s_idx)
            if ed <= K: continue

            y_now = y_tokens.clone(); y_now[:, ed:] = -1
            parts = list(torch.split(torch.clamp(y_now, min=0), lens, dim=1))
            x_cvae = vae.quantize.idxBl_to_var_input(parts)

            logits_full = m(lab, x_cvae)
            Lp = logits_full.size(1)
            j0 = max(0, min(st, Lp-1))
            j1 = max(0, min(ed, Lp))

            temp = float(lin(TEMP_MIN, TEMP_MAX, s_idx))
            topk = int(round(lin(TOPK_MIN, TOPK_MAX, s_idx)))
            topp = float(lin(TOPP_MIN, TOPP_MAX, s_idx))

            for pos in range(max(j0, K), j1):
                logits_t = logits_full[:, pos, :] - uni_bias.view(1, -1)
                if REPEAT_PENALTY and REPEAT_PENALTY > 1.0:
                    window = y_tokens[:, max(0, pos-REPEAT_WINDOW):pos]
                    for b in range(B):
                        uniq = torch.unique(window[b][window[b] >= 0])
                        if len(uniq) > 0: logits_t[b, uniq] /= REPEAT_PENALTY
                if topk > 0:
                    k = min(topk, logits_t.size(-1))
                    kth = torch.topk(logits_t, k=k, dim=-1).values[..., -1,None]
                    logits_t = torch.where(logits_t < kth, torch.full_like(logits_t, float('-inf')), logits_t)
                if 0.0 < topp < 1.0:
                    sorted_logits, idx = torch.sort(logits_t, descending=True, dim=-1)
                    probs_sorted = torch.softmax(sorted_logits / max(1e-6, temp), dim=-1)
                    keep = probs_sorted.cumsum(-1) <= topp
                    keep[...,0] = True
                    sorted_logits = sorted_logits.masked_fill(~keep, float('-inf'))
                    logits_t = torch.full_like(logits_t, float('-inf')).scatter(-1, idx, sorted_logits)
                tok = torch.multinomial(torch.softmax(logits_t / max(1e-6, temp), dim=-1), 1).squeeze(1)
                y_tokens[:, pos] = tok

        # Decode tokens -> image
        if hasattr(vae, "decode_tokens") and callable(getattr(vae, "decode_tokens")):
            imgs = vae.decode_tokens(y_tokens)
        else:
            imgs = vae.idxBl_to_img(list(torch.split(y_tokens, lens, dim=1)), same_shape=True, last_one=True)

        for i, img in enumerate(imgs):
            cls_name = class_names[cid]
            fn = out_dir / f"ep{ep:03d}_cls{cid}_{cls_name}_K{K}_{'TP' if use_teacher else 'NOP'}_{i:02d}.png"
            to_pil(img.clamp(0,1).cpu()).save(str(fn))
    m.train(prev)

# ====== plots & logs ======
def _plot_one(x, ys, labels, title, save_path):
    plt.figure(); plt.grid(True, alpha=0.3)
    for y, lb in zip(ys, labels):
        plt.plot(x, y, label=lb)
    plt.xlabel("epoch"); plt.ylabel(title); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()

def save_plots(ep_hist, trLm, trLt, trAm, trAt, vaLm, vaLt, vaAm, vaAt, out_dir):
    xs = ep_hist
    _plot_one(xs, [trLm, vaLm], ["train L_mean", "val L_mean"], "Cross-Entropy (mean)", out_dir/"curve_Lmean.png")
    _plot_one(xs, [trLt, vaLt], ["train L_tail", "val L_tail"], "Cross-Entropy (tail)", out_dir/"curve_Ltail.png")
    _plot_one(xs, [trAm, vaAm], ["train Acc_mean", "val Acc_mean"], "Token accuracy (mean, %)", out_dir/"curve_Acc_mean.png")
    _plot_one(xs, [trAt, vaAt], ["train Acc_tail", "val Acc_tail"], "Token accuracy (tail, %)", out_dir/"curve_Acc_tail.png")

def save_metrics_csv(ep_hist, trLm, trLt, trAm, trAt, vaLm, vaLt, vaAm, vaAt, lrs, best_ep):
    with open(METRICS_DIR/"metrics_online.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","train_Lm","train_Lt","train_Am","train_At","vL_mean","vL_tail","vacc_mean","vacc_tail","lr"])
        for i, ep in enumerate(ep_hist):
            w.writerow([
                int(ep),
                float(trLm[i]) if i < len(trLm) else "",
                float(trLt[i]) if i < len(trLt) else "",
                float(trAm[i]) if i < len(trAm) else "",
                float(trAt[i]) if i < len(trAt) else "",
                float(vaLm[i]) if i < len(vaLm) else "",
                float(vaLt[i]) if i < len(vaLt) else "",
                float(vaAm[i]) if i < len(vaAm) else "",
                float(vaAt[i]) if i < len(vaAt) else "",
                float(lrs[i]) if i < len(lrs) else "",
            ])
    with open(METRICS_DIR/"best.txt","w") as f:
        f.write(str(best_ep))

# ====== Export 'sampling_pack' (robust) ======
def _lens_from_vnums(vnums): return [int(v)*int(v) for v in vnums]

@torch.no_grad()
def dump_sampling_pack(vae: VQVAE, train_root: str, classes: list, out_dir: Path,
                       vocab_size: int = VOCAB_SIZE, tp_bank_max: int = 0):
    """
    Robust exporter: skips bad images, always dumps teacher_bank if tp_bank_max >= 0.
    """
    pack = out_dir / "sampling_pack"
    pack.mkdir(parents=True, exist_ok=True)

    # fixed classes & vnums
    (pack / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    vnums = tuple(int(x) for x in vae.quantize.v_patch_nums)
    (pack / "vnums.json").write_text(json.dumps(list(vnums)))

    lens = _lens_from_vnums(vnums)
    Ltot = int(sum(lens)); C = len(classes); V = int(vocab_size)
    K_edges = np.cumsum([0] + lens)[1:].tolist()

    # dataset meta (no aug)
    ds = datasets.ImageFolder(train_root)
    assert ds.classes == classes, "ImageFolder 类别顺序与训练时不一致！"

    idx_all, lab_all = [], []
    counts  = np.full((C, Ltot, V), 0.5, dtype=np.float64)
    teacher_pref = np.zeros((C, Ltot), dtype=np.int64)
    seen_first   = [False]*C
    uni_cnt = np.zeros((V,), dtype=np.int64)
    bank_lists = [[] for _ in range(C)]
    per_scale = {v: [] for v in vnums}

    kept = 0; skipped = 0
    for path, lab in ds.samples:
        try:
            pil = _default_loader(path)     # may raise
            img = NOAUG_TF(pil).unsqueeze(0).to(DEVICE)
            idxBl_list = vae.img_to_idxBl(img)  # list of (1, v*v)
            idx1d = torch.cat(idxBl_list, dim=1)[0].cpu().numpy().astype(np.int64)  # (Ltot,)
        except Exception as e:
            skipped += 1
            print(f"[sampling-pack] skip bad: {path} | {e}")
            continue

        idx_all.append(idx1d); lab_all.append(int(lab))
        for v, blk in zip(vnums, idxBl_list):
            per_scale[v].append(blk.view(v, v)[0].cpu().numpy().astype(np.int64))

        c = int(lab)
        if not seen_first[c]:
            teacher_pref[c] = idx1d; seen_first[c] = True
        counts[c, np.arange(Ltot), idx1d] += 1.0
        uni_cnt += np.bincount(idx1d, minlength=V)
        # bank collect (0 == no limit)
        if (tp_bank_max == 0) or (len(bank_lists[c]) < tp_bank_max):
            bank_lists[c].append(idx1d)
        kept += 1

    print(f"[sampling-pack] kept={kept}, skipped={skipped}")

    # tokens_multiscale_maps.npz
    data = {"vnums": np.asarray(vnums, np.int32),
            "idx":   np.stack(idx_all, axis=0).astype(np.int64),
            "label": np.asarray(lab_all, dtype=np.int64)}
    for v in vnums:
        data[f"tok_{v}x{v}"] = np.stack(per_scale[v], axis=0).astype(np.int64)
    np.savez_compressed(pack / "tokens_multiscale_maps.npz", **data)

    # stats_full.npz
    counts /= counts.sum(axis=-1, keepdims=True)
    uni_prob = uni_cnt / max(1.0, uni_cnt.sum())
    uni_bias = UNIGRAM_ALPHA * np.log(uni_prob + 1e-8)
    np.savez_compressed(
        pack / "stats_full.npz",
        pos_prior=counts.astype(np.float32),
        teacher_pref=teacher_pref.astype(np.int64),
        uni_bias=uni_bias.astype(np.float32),
        vnums=np.asarray(vnums, np.int64),
        lens=np.asarray(lens, np.int64)
    )

    # teacher_bank.npz  (>=0 → always save)
    if tp_bank_max >= 0:
        bank_flat = np.concatenate(
            [np.stack(b, 0) if len(b)>0 else np.zeros((1, Ltot), np.int64) for b in bank_lists],
            axis=0
        )
        st = 0; spans=[]
        for b in bank_lists:
            n = max(1, len(b)); spans.append((st, st+n)); st += n
        np.savez_compressed(pack / "teacher_bank.npz",
                            bank=bank_flat,
                            spans=np.asarray(spans, np.int64),
                            vnums=np.asarray(vnums, np.int64),
                            lens=np.asarray(lens, np.int64))

    # meta.json
    meta = dict(L_total=Ltot, vocab=V, C=C, K_edges=K_edges)
    (pack / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[sampling-pack] done → {pack}")

# ====== Main ======
def main():
    print(f"\n[setup] device={DEVICE}, INPUT_RESO={INPUT_RESO}\n")

    # Data
    tr_dl, tr_ds = build_loader(DATA_ROOT_TRAIN, train_tf, BATCH, True)
    va_dl, va_ds = build_loader(DATA_ROOT_VAL,   val_tf,   BATCH, False)
    NUM_CLASS = len(tr_ds.classes)
    # per-class counts
    if hasattr(tr_ds, "targets"):
        from collections import Counter
        cnt = Counter(tr_ds.targets)
        human_cnt = {tr_ds.classes[i]: int(cnt.get(i,0)) for i in range(NUM_CLASS)}
        print("[data] per-class counts:", human_cnt)
    print(f"[data] classes={NUM_CLASS}, names={tr_ds.classes}")

    # save class order for alignment
    with open(OUT_DIR/"classes.json","w") as f:
        json.dump(tr_ds.classes, f, ensure_ascii=False, indent=2)

    # VQVAE (frozen)
    vae = VQVAE(
        vocab_size=VOCAB_SIZE, z_channels=16, ch=64, beta=0.25,
        test_mode=True, v_patch_nums=(1,2,3,4,5,6,8,10,13,16)
    ).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location="cpu"), strict=False)
    with torch.no_grad():
        vae.quantize.embedding.weight.data.copy_(torch.load(CODEBOOK_PTH, map_location="cpu"))
    vae.eval(); [p.requires_grad_(False) for p in vae.parameters()]
    stride = _infer_total_stride(vae, INPUT_RESO)
    print(f"[vae] total stride inferred = {stride}, v_patch_nums = {tuple(int(x) for x in vae.quantize.v_patch_nums)}")

    # tokens length metadata
    VNUMS = tuple(int(x) for x in vae.quantize.v_patch_nums)
    lens, L_TOTAL, spans = _split_lens_from_vnums(VNUMS)
    LAST_L = lens[-1]
    print(f"[tokens] VNUMS={VNUMS}, L_TOTAL={L_TOTAL}, LAST_L={LAST_L}, S={len(lens)}\n")

    # VAR
    model = VAR(
        vae_local=vae, num_classes=NUM_CLASS,
        depth=DEPTH, embed_dim=DIM, num_heads=HEADS,
        patch_nums=VNUMS,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        flash_if_available=False, fused_if_available=False
    ).to(DEVICE)
    model.init_weights()

    # optimizer
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith('bias') or ('norm' in n.lower()) or ('embedding' in n.lower()):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {'params': decay,    'weight_decay': 1e-3},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    opt = torch.optim.AdamW(param_groups, lr=LR, betas=(0.9, 0.95))

    # scheduler
    if USE_COSINE:
        scheduler = CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=MIN_LR)

    # history
    EP_hist=[]; TR_LM=[]; TR_LT=[]; TR_AM=[]; TR_AT=[]
    VA_LM=[]; VA_LT=[]; VA_AM=[]; VA_AT=[]; LR_hist=[]
    BEST_EPOCH=-1; best_vL_mean=float("inf")

    try:
        # loop
        for ep in range(1, EPOCHS+1):
            # warmup / cosine
            if USE_COSINE:
                if ep <= WARMUP_EPOCHS:
                    warm_lr = LR * ep / max(1, WARMUP_EPOCHS)
                    for g in opt.param_groups: g['lr'] = warm_lr
                else:
                    scheduler.step()

            # close label smoothing in last 30 epochs
            if ep == EPOCHS - 30 + 1:
                global ce_train_nored
                ce_train_nored = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')

            t0=time.time()
            trLm, trLt, trAm, trAt = run_epoch_train(model, vae, tr_dl, lens, L_TOTAL, spans, ep, EPOCHS, opt)

            need_eval = (ep % EVAL_PERIOD == 0) or (ep == EPOCHS)
            if need_eval:
                vL_mean, vL_tail, vAm, vAt = evaluate(model, vae, va_dl, lens, L_TOTAL)
                if vL_mean < best_vL_mean:
                    best_vL_mean = vL_mean; BEST_EPOCH = ep
                    torch.save(model.state_dict(), OUT_DIR/"ar-ckpt-best.pth")
            else:
                vL_mean=vL_tail=vAm=vAt=float("nan")

            if (ep % SAVE_EVERY == 0) or (ep == EPOCHS):
                torch.save(model.state_dict(), OUT_DIR/"ar-ckpt-last.pth")

            cur_lr = opt.param_groups[0]['lr']
            print(f" [*] [ep{ep:03d}]  train Lm/Lt={trLm:.4f}/{trLt:.4f}  "
                  f"val Lm/Lt={vL_mean:.4f}/{vL_tail:.4f}  "
                  f"Acc m&t: {trAm:.2f}% {trAt:.2f}% ({vAm:.2f}% {vAt:.2f}%)  "
                  f"lr {cur_lr:.2e}  ({time.time()-t0:.1f}s)")

            EP_hist.append(ep)
            TR_LM.append(trLm); TR_LT.append(trLt); TR_AM.append(trAm); TR_AT.append(trAt)
            VA_LM.append(vL_mean); VA_LT.append(vL_tail); VA_AM.append(vAm); VA_AT.append(vAt)
            LR_hist.append(cur_lr)

            # save CSV & curves
            save_metrics_csv(EP_hist, TR_LM, TR_LT, TR_AM, TR_AT, VA_LM, VA_LT, VA_AM, VA_AT, LR_hist, BEST_EPOCH)
            save_plots(EP_hist, TR_LM, TR_LT, TR_AM, TR_AT, VA_LM, VA_LT, VA_AM, VA_AT, METRICS_DIR)

            # quicklook sampling
            if DO_SAMPLING and (ep in SAMPLE_EARLY_EPOCHS or (ep % SAMPLE_EVERY == 0)):
                maybe_sample_and_save(ep, vae, model, DATA_ROOT_TRAIN, tr_ds.classes,
                                      SAMPLES_DIR, num_per_class=PER_CLASS_SAMPLE,
                                      coarse_K=COARSE_K, use_teacher=USE_TEACHER_PREFIX)

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("\n✅ Training finished →", OUT_DIR)
        print(f"✅ Best epoch by val L_mean: {BEST_EPOCH}")

    finally:
        # export sampling pack (robust)
        if SAVE_SAMPLING_PACK:
            try:
                dump_sampling_pack(vae, DATA_ROOT_TRAIN, tr_ds.classes, OUT_DIR,
                                   vocab_size=VOCAB_SIZE, tp_bank_max=TP_BANK_MAX)
            except Exception as e:
                print(f"[sampling-pack] skipped due to error: {e}")

        # clean up
        plt.close('all')
        del tr_dl, va_dl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
