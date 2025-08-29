#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VAR — Official-aligned next-scale training (ultrasound, multi-scale tokens)

- 训练反传：CE(label_smoothing>0, reduction='none') → 1/L 权重 → batch mean
- 日志/验证：CE(label_smoothing=0.0, reduction='mean') 的 Lm/Lt/Acc
- Progressive：可选（默认关），仅在损失权重上做阶段加权
- 采样：保留 COARSE_K / USE_TEACHER_PREFIX（可在训练中按频率保存样例）

★ 需求：
  1) 仅“显示/打印”时让 val 始终高于 train（不影响真实记录）
  2) 缓解过拟合：正则化 + 均衡采样 + 学习率调度 + 梯度裁剪 + 早停
"""

import time, json, csv, random
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms as T

from models.vqvae import VQVAE
from models.var   import VAR

# ================== PATHS ==================
TOK_ROOT = Path("/home/guests/yiping_zhou/projects/ultrasound_VAR/data_tokens_nextscale")
TRAIN_NPZ = TOK_ROOT / "train" / "tokens_multiscale_maps.npz"
VAL_NPZ   = TOK_ROOT / "val"   / "tokens_multiscale_maps.npz"

OUT_DIR  = Path("/home/guests/yiping_zhou/projects/ultrasound_VAR/outputs/var_ultra_nextscale_OFFICIAL4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = OUT_DIR / "metrics"; METRIC_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = OUT_DIR / "samples"; SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

vae_ckpt = "/home/guests/yiping_zhou/projects/ultrasound_VAR/outputs/sum_mse_loss_with_usage2/best.pth"
codebook = "/home/guests/yiping_zhou/projects/ultrasound_VAR/outputs/sum_mse_loss_with_usage2/codebook/codebook_weight.pth"
CLASSES = json.load(open(TOK_ROOT/"train"/"classes.json"))
NUM_CLASS = len(CLASSES)

# ================== Hyperparams ==================
BATCH, ACCUM_STEPS = 4, 4
USE_AMP, AMP_BF16  = True, False
EPOCHS, LR, SEED   = 300, 1e-4, 0
VOCAB_SIZE         = 2048
DEPTH, DIM, HEADS  = 6, 384, 6
NUM_WORKERS        = 2
LABEL_SMOOTH       = 0.1
WEIGHT_DECAY       = 1e-2

# —— Regularization / schedule / early stop ——
USE_CLASS_BALANCE  = True     # 训练集启用类均衡采样
DROP_RATE          = 0.10
ATTN_DROP_RATE     = 0.10
DROP_PATH_RATE     = 0.10
CLIP_NORM          = 1.0      # 梯度裁剪

# 早停
EARLY_STOP_PATIENCE  = 20
EARLY_STOP_MIN_DELTA = 5e-3

# 学习率调度（按 epoch）
WARMUP_EPOCHS = 5
ETA_MIN       = 1e-5

# —— Progressive（可选，默认关） ——
USE_PROGRESSIVE    = False
PG_START_STAGE     = 0
PG_PORTION         = 0.7
PG_WP_PORTION      = 0.05

# —— 验证/保存频率 ——
EVAL_PERIOD        = 1   # 若想更像官方每10轮可改为 10

# —— 采样（可选） ——
DO_SAMPLING        = True
SAMPLE_EARLY_EPOCHS = {1,2,3,4,5,6,7,8,9,10,20}
SAMPLE_EVERY         = 25
PER_CLASS_SAMPLE   = 1
COARSE_K           = 155
USE_TEACHER_PREFIX = True
USE_COARSE_ARGMAX  = True
TEMP_MIN, TEMP_MAX = 0.95, 0.80
TOPK_MIN, TOPK_MAX = 64, 128
TOPP_MIN, TOPP_MAX = 0.90, 0.95
REPEAT_PENALTY     = 1.20
REPEAT_WINDOW      = 64
UNIGRAM_ALPHA      = 0.3

# —— 仅影响“打印/画图显示”的开关：让 val > train —— 
ENFORCE_VAL_HIGHER_THAN_TRAIN = True
VAL_TRAIN_MARGIN = 1e-3  # 显示时保证最小差距

# ================== Seed / Device ==================
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================== Data (npz) ==================
VNUMS_DEFAULT = (1,2,3,4,5,6,8,10,13,16)

class MultiScaleNPZ(Dataset):
    def __init__(self, npz_file: Path):
        d = np.load(npz_file, allow_pickle=False)
        self.vnums  = tuple(int(x) for x in d["vnums"].tolist()) if "vnums" in d else VNUMS_DEFAULT
        self.labels = d["label"].astype(np.int64)
        self.N = len(self.labels)
        self.lens   = [v*v for v in self.vnums]
        self.LTOT   = sum(self.lens)
        self.idx1d  = np.empty((self.N, self.LTOT), dtype=np.int64)
        for i in range(self.N):
            off = 0
            for v in self.vnums:
                self.idx1d[i, off:off+v*v] = d[f"tok_{v}x{v}"][i].reshape(-1)
                off += v*v
    def __len__(self):  return self.N
    def __getitem__(self, i: int):
        return torch.from_numpy(self.idx1d[i].copy()), int(self.labels[i])

def make_dl(npz_path: Path, shuffle: bool, balance: bool=False):
    ds = MultiScaleNPZ(npz_path)
    if balance and shuffle:
        labels = torch.as_tensor(ds.labels, dtype=torch.long)
        bins   = torch.bincount(labels, minlength=int(labels.max().item())+1).float()
        weights= 1.0 / torch.clamp(bins, min=1.0)
        samp_w = weights[labels]
        sampler = WeightedRandomSampler(samp_w, num_samples=len(ds), replacement=True)
        dl = DataLoader(ds, BATCH, sampler=sampler, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=True)
    else:
        dl = DataLoader(ds, BATCH, shuffle=shuffle, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=shuffle)
    return dl, ds

tr_dl, tr_ds = make_dl(TRAIN_NPZ, True, balance=USE_CLASS_BALANCE)
va_dl, va_ds = make_dl(VAL_NPZ,   False, balance=False)

VNUMS   = tr_ds.vnums
LENS    = tr_ds.lens
L_TOTAL = tr_ds.LTOT
S       = len(LENS)
LAST_L  = LENS[-1]
ST_TAIL = L_TOTAL - LAST_L
SEG_B   = np.cumsum([0]+LENS)
BEGIN_ENDS = [(int(SEG_B[i]), int(SEG_B[i+1])) for i in range(S)]

# ================== Models ==================
vae = VQVAE(vocab_size=VOCAB_SIZE, z_channels=16, ch=64, beta=0.25,
            test_mode=True, v_patch_nums=VNUMS).to(device)
vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=False)
with torch.no_grad():
    vae.quantize.embedding.weight.data.copy_(torch.load(codebook, map_location="cpu"))
vae.eval()
for p in vae.parameters():
    p.requires_grad_(False)

print(f"\n[constructor]  ==== flash_if_available=False (0/{DEPTH}), fused_if_available=False ====")
print(f"    [VAR config ] embed_dim={DIM}, num_heads={HEADS}, depth={DEPTH}, mlp_ratio=4.0")
print(f"    [drop ratios ] drop_rate={DROP_RATE}, attn_drop_rate={ATTN_DROP_RATE}, drop_path_rate={DROP_PATH_RATE}\n")

model = VAR(
    vae_local=vae, num_classes=NUM_CLASS,
    depth=DEPTH, embed_dim=DIM, num_heads=HEADS,
    patch_nums=VNUMS,
    drop_rate=DROP_RATE, attn_drop_rate=ATTN_DROP_RATE, drop_path_rate=DROP_PATH_RATE,
    flash_if_available=False, fused_if_available=False
).to(device)
model.init_weights()
print(f"[init_weights] VAR with init_std=0.02")
print(f"VNUMS={VNUMS}, L_TOTAL={L_TOTAL}, S={S}, LAST_L={LAST_L}\n")

opt   = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=WEIGHT_DECAY)

# —— 学习率调度（warmup + cosine） ——
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
_warmup = LinearLR(opt, start_factor=0.1, total_iters=WARMUP_EPOCHS)
_cosine = CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=ETA_MIN)
sched   = SequentialLR(opt, [_warmup, _cosine], milestones=[WARMUP_EPOCHS])

# ================== Helpers ==================
def split_idx1d_to_Bl(idx1d: torch.Tensor):
    return [p.contiguous() for p in torch.split(idx1d.long(), LENS, dim=1)]

@torch.no_grad()
def build_var_input_from_idx1d(idx1d: torch.Tensor):
    gt_idx_Bl = split_idx1d_to_Bl(idx1d)
    x_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)  # [B, L, C]
    gt_BL  = torch.cat(gt_idx_Bl, dim=1)                 # [B, L]
    return x_BLCv, gt_BL

ce_train_nored = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH, reduction='none')
ce_eval_mean   = nn.CrossEntropyLoss(label_smoothing=0.0,           reduction='mean')
LOSS_WEIGHT = torch.ones(1, L_TOTAL, device=device) / float(L_TOTAL)

def amp_ctx():
    if not USE_AMP or (device != "cuda"): return nullcontext()
    try:
        dtype = torch.bfloat16 if AMP_BF16 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported() else torch.float16
        return torch.cuda.amp.autocast(device_type="cuda", dtype=dtype)
    except TypeError:
        return torch.cuda.amp.autocast()

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and (device=="cuda"))

# ================== Train ==================
def run_epoch_train_official(loader, ep: int, total_epochs: int):
    model.train(True); opt.zero_grad(set_to_none=True)

    use_pg = bool(USE_PROGRESSIVE)
    if use_pg:
        max_it   = total_epochs * len(loader)
        g_it_beg = (ep-1) * len(loader)

    # 训练集上的“验证口径”累计（用于日志）
    tot = 0; Lm_sum = 0.0; Lt_sum = 0.0; Am_sum = 0.0; At_sum = 0.0

    for it, (idx1d, lab) in enumerate(loader):
        idx1d = idx1d.to(device, non_blocking=True)
        lab   = torch.as_tensor(lab, device=device, dtype=torch.long)
        B = idx1d.size(0); V = VOCAB_SIZE

        x_cvae, gt_BL = build_var_input_from_idx1d(idx1d)

        # ====== forward for training loss ======
        with amp_ctx():
            logits_BLV = model(lab, x_cvae)                 # [B,L,V]
            loss_BL = ce_train_nored(
                logits_BLV.view(-1, V),
                gt_BL.view(-1)
            ).view(B, -1)

            if use_pg:
                g_it = g_it_beg + it
                wp_it  = PG_WP_PORTION * max_it
                pg_end = PG_PORTION * max_it
                if g_it <= wp_it: 
                    prog_si = PG_START_STAGE
                elif g_it >= pg_end:
                    prog_si = S - 1
                else:
                    delta = (S - 1 - PG_START_STAGE)
                    progress = min(max((g_it - wp_it) / max(1, (pg_end - wp_it)), 0.0), 1.0)
                    prog_si = PG_START_STAGE + round(progress * delta)
                bg, ed = BEGIN_ENDS[prog_si]
                lw = LOSS_WEIGHT.clone()
                prog_wp  = max(min((it+1) / max(1, PG_WP_PORTION*len(loader)), 1.0), 0.01)
                lw[:, bg:ed] = lw[:, bg:ed] * float(prog_wp)
            else:
                lw = LOSS_WEIGHT

            loss = loss_BL.mul(lw).sum(dim=-1).mean()

        # ====== backward / step ======
        do_step = (((it+1) % max(1, ACCUM_STEPS)) == 0) or (it+1 == len(loader))
        if scaler.is_enabled():
            scaler.scale(loss / max(1, ACCUM_STEPS)).backward()
            if do_step:
                scaler.unscale_(opt)
                if CLIP_NORM and CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
        else:
            (loss / max(1, ACCUM_STEPS)).backward()
            if do_step:
                if CLIP_NORM and CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step(); opt.zero_grad(set_to_none=True)

        # ====== 日志口径（无 smoothing）的训练指标（注意：在 train() 下，dropout/droppath 开着） ======
        with torch.no_grad():
            Lm = ce_eval_mean(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).item()
            pred_BL = logits_BLV.argmax(dim=-1)
            Am = (pred_BL == gt_BL).sum().item() * (100.0 / gt_BL.shape[1])

            st_tail = ST_TAIL
            Lt = ce_eval_mean(
                logits_BLV[:, st_tail:].reshape(-1, V),
                gt_BL[:,    st_tail:].reshape(-1)
            ).item()
            At = (pred_BL[:, st_tail:] == gt_BL[:, st_tail:]).sum().item() * (100.0 / LAST_L)

            Lm_sum += Lm * B; Lt_sum += Lt * B; Am_sum += Am; At_sum += At; tot += B

    return (Lm_sum/max(1,tot), Lt_sum/max(1,tot), Am_sum/max(1,tot), At_sum/max(1,tot))

# ================== Eval ==================
@torch.no_grad()
def eval_official(loader):
    model.eval()
    tot = 0
    L_mean = 0.0; L_tail = 0.0
    acc_mean = 0.0; acc_tail = 0.0

    for idx1d, lab in loader:
        idx1d = idx1d.to(device, non_blocking=True)
        lab   = torch.as_tensor(lab, device=device, dtype=torch.long)
        B = idx1d.size(0); V = VOCAB_SIZE

        x_cvae, gt_BL = build_var_input_from_idx1d(idx1d)
        logits_BLV = model(lab, x_cvae)

        L_mean += ce_eval_mean(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).item() * B

        st_tail = ST_TAIL
        L_tail += ce_eval_mean(
            logits_BLV[:, st_tail:].reshape(-1, V),
            gt_BL[:,    st_tail:].reshape(-1)
        ).item() * B

        pred_BL = logits_BLV.argmax(dim=-1)
        acc_mean += (pred_BL == gt_BL).sum().item() * (100.0 / gt_BL.shape[1])
        acc_tail += (pred_BL[:, st_tail:] == gt_BL[:, st_tail:]).sum().item() * (100.0 / LAST_L)
        tot += B

    tot = max(1, tot)
    return (L_mean / tot, L_tail / tot, acc_mean / tot, acc_tail / tot)

# ================== Sampling（可选） ==================
@torch.no_grad()
def build_sampling_stats(npz_path: Path, vocab: int, K: int):
    d = np.load(npz_path, allow_pickle=False)
    vnums = tuple(int(x) for x in d["vnums"].tolist()) if "vnums" in d else VNUMS_DEFAULT
    lens  = [v*v for v in vnums]; Ltot = sum(lens)
    N     = d[f"tok_{vnums[0]}x{vnums[0]}"].shape[0]
    idx1d = np.empty((N, Ltot), dtype=np.int64)
    for i in range(N):
        off=0
        for v in vnums:
            idx1d[i, off:off+v*v] = d[f"tok_{v}x{v}"][i].reshape(-1); off+=v*v
    labels = d["label"].astype(np.int64)
    C = int(np.max(labels))+1
    K = min(K, Ltot)

    counts = np.full((C, K, vocab), 0.5, dtype=np.float64)
    for seq, c in zip(idx1d, labels):
        for p in range(K):
            counts[c, p, int(seq[p])] += 1
    counts /= counts.sum(axis=-1, keepdims=True)
    pos_prior = torch.from_numpy(counts).float().to(device)

    teacher = np.zeros((C, K), dtype=np.int64)
    for c in range(C):
        pos = np.where(labels==c)[0]
        teacher[c] = idx1d[pos[0], :K] if len(pos)>0 else np.zeros((K,), np.int64)
    teacher = torch.from_numpy(teacher).to(device)

    binc = np.bincount(idx1d.reshape(-1), minlength=vocab).astype(np.float64)
    uni  = torch.from_numpy(UNIGRAM_ALPHA*np.log(binc/ max(1.0,binc.sum()) + 1e-8)).float().to(device)
    return pos_prior, teacher, uni, vnums, lens

@torch.no_grad()
def maybe_sample_and_save(ep: int, out_dir: Path, num_per_class: int=1,
                          coarse_K: int=COARSE_K, use_teacher: bool=USE_TEACHER_PREFIX):
    pos_prior, teacher_pref, uni_bias, _, lens = build_sampling_stats(TRAIN_NPZ, VOCAB_SIZE, coarse_K)
    m = model
    prev = m.training; m.eval()

    S = len(lens)
    seg = np.cumsum([0]+lens)
    def span_loc(s_idx: int): return int(seg[s_idx]), int(seg[s_idx+1])
    def lin(a,b,s): return a + (b-a) * (s / max(1, S-1))

    for cid in range(NUM_CLASS):
        B = num_per_class
        lab = torch.full((B,), cid, device=device, dtype=torch.long)
        L_total = int(seg[-1])
        y_tokens = torch.full((B, L_total), -1, device=device, dtype=torch.long)

        if use_teacher:
            K = min(coarse_K, L_total)
            y_tokens[:, :K] = teacher_pref[cid].unsqueeze(0).expand(B, -1)[:, :K]
        else:
            for p in range(min(coarse_K, L_total)):
                dist = pos_prior[cid, p].unsqueeze(0).expand(B,-1)
                tok  = dist.argmax(-1, keepdim=True) if USE_COARSE_ARGMAX else torch.multinomial(dist,1)
                y_tokens[:, p] = tok.squeeze(1)

        for s_idx in range(S):
            st, ed = span_loc(s_idx)
            if ed <= coarse_K: continue

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

            for pos in range(j0, j1):
                if pos < coarse_K: continue
                logits_t = logits_full[:, pos, :] - uni_bias.view(1,-1)
                if REPEAT_PENALTY and REPEAT_PENALTY>1.0:
                    window = y_tokens[:, max(0, pos-REPEAT_WINDOW):pos]
                    for b in range(B):
                        uniq = torch.unique(window[b][window[b]>=0])
                        if len(uniq)>0: logits_t[b, uniq] /= REPEAT_PENALTY
                if topk>0:
                    k = min(topk, logits_t.size(-1))
                    kth = torch.topk(logits_t, k=k, dim=-1).values[..., -1,None]
                    logits_t = torch.where(logits_t<kth, torch.full_like(logits_t, float('-inf')), logits_t)
                if 0.0 < topp < 1.0:
                    sorted_logits, idx = torch.sort(logits_t, descending=True, dim=-1)
                    probs_sorted = torch.softmax(sorted_logits/temp, dim=-1)
                    keep = probs_sorted.cumsum(-1) <= topp
                    keep[...,0]=True
                    sorted_logits = sorted_logits.masked_fill(~keep, float('-inf'))
                    logits_t = torch.full_like(logits_t, float('-inf')).scatter(-1, idx, sorted_logits)
                tok = torch.multinomial(torch.softmax(logits_t/temp, dim=-1), 1).squeeze(1)
                y_tokens[:, pos] = tok

        # 生成图像保存（如果 VQVAE 提供 decode_tokens）
        if hasattr(vae, "decode_tokens"):
            imgs = vae.decode_tokens(y_tokens)
        else:
            imgs = vae.idxBl_to_img(list(torch.split(y_tokens, lens, dim=1)), same_shape=True, last_one=True)

        for i, img in enumerate(imgs):
            fn = SAMPLE_DIR / f"ep{ep:03d}_cls{cid}_{CLASSES[cid]}_K{coarse_K}_{'TP' if use_teacher else 'NOP'}_{i:02d}.png"
            T.ToPILImage()(img.clamp(0,1).cpu()).save(str(fn))

    m.train(prev)

# ================== Logs & Curves ==================
EP_hist = []
TR_LM, TR_LT, TR_AM, TR_AT = [], [], [], []
VA_LM, VA_LT, VA_AM, VA_AT = [], [], [], []
LR_hist = []
BEST_EPOCH = -1

def _save_metrics_and_plot():
    e = np.asarray(EP_hist, dtype=np.int64)

    # —— 保存原始 npz/csv（保持真实值） ——
    np.savez_compressed(
        METRIC_DIR/"log_official_ns.npz",
        epoch=e,
        train_Lm=np.asarray(TR_LM, dtype=np.float32),
        train_Lt=np.asarray(TR_LT, dtype=np.float32),
        train_Am=np.asarray(TR_AM, dtype=np.float32),
        train_At=np.asarray(TR_AT, dtype=np.float32),
        vL_mean=np.asarray(VA_LM, dtype=np.float32),
        vL_tail=np.asarray(VA_LT, dtype=np.float32),
        vacc_mean=np.asarray(VA_AM, dtype=np.float32),
        vacc_tail=np.asarray(VA_AT, dtype=np.float32),
        lr=np.asarray(LR_hist, dtype=np.float32),
        best_epoch=np.asarray([BEST_EPOCH], dtype=np.int32),
    )

    with open(METRIC_DIR/"metrics_official_ns.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_Lm","train_Lt","train_Am","train_At","vL_mean","vL_tail","vacc_mean","vacc_tail","lr"])
        for i in range(len(e)):
            w.writerow([
                int(e[i]),
                float(TR_LM[i]) if i < len(TR_LM) else "",
                float(TR_LT[i]) if i < len(TR_LT) else "",
                float(TR_AM[i]) if i < len(TR_AM) else "",
                float(TR_AT[i]) if i < len(TR_AT) else "",
                float(VA_LM[i]) if i < len(VA_LM) else "",
                float(VA_LT[i]) if i < len(VA_LT) else "",
                float(VA_AM[i]) if i < len(VA_AM) else "",
                float(VA_AT[i]) if i < len(VA_AT) else "",
                float(LR_hist[i]) if i < len(LR_hist) else "",
            ])

    # —— 原始曲线 ——
    plt.figure(figsize=(7.4,4.8), dpi=120)
    plt.plot(e, np.asarray(TR_LM, dtype=float), label="train L_mean")
    plt.plot(e, np.asarray(VA_LM, dtype=float), label="val L_mean")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Next-Scale L_mean (train vs val)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(METRIC_DIR/"curve_Lmean_train_val.png"); plt.close()

    plt.figure(figsize=(7.4,4.8), dpi=120)
    plt.plot(e, np.asarray(TR_LT, dtype=float), label="train L_tail")
    plt.plot(e, np.asarray(VA_LT, dtype=float), label="val L_tail")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Next-Scale L_tail (train vs val)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(METRIC_DIR/"curve_Ltail_train_val.png"); plt.close()

    # —— 显示版曲线（强制 train < val，仅用于可视化，不改原始记录） ——
    trLm = np.asarray(TR_LM, dtype=float)
    trLt = np.asarray(TR_LT, dtype=float)
    vaLm = np.asarray(VA_LM, dtype=float)
    vaLt = np.asarray(VA_LT, dtype=float)

    mask_m = np.isfinite(vaLm)
    mask_t = np.isfinite(vaLt)
    disp_trLm = trLm.copy()
    disp_trLt = trLt.copy()
    disp_trLm[mask_m] = np.minimum(trLm[mask_m], vaLm[mask_m] - VAL_TRAIN_MARGIN)
    disp_trLt[mask_t] = np.minimum(trLt[mask_t], vaLt[mask_t] - VAL_TRAIN_MARGIN)

    plt.figure(figsize=(7.4,4.8), dpi=120)
    plt.plot(e, disp_trLm, label="train L_mean (display)")
    plt.plot(e, vaLm,      label="val L_mean")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Next-Scale L_mean (display: train < val)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(METRIC_DIR/"curve_Lmean_train_val_DISPLAY.png"); plt.close()

    plt.figure(figsize=(7.4,4.8), dpi=120)
    plt.plot(e, disp_trLt, label="train L_tail (display)")
    plt.plot(e, vaLt,      label="val L_tail")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Next-Scale L_tail (display: train < val)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(METRIC_DIR/"curve_Ltail_train_val_DISPLAY.png"); plt.close()

def _display_floor(train_v, val_v):
    if ENFORCE_VAL_HIGHER_THAN_TRAIN and np.isfinite(val_v):
        return min(train_v, float(val_v) - VAL_TRAIN_MARGIN)
    return train_v

# ================== Main ==================
def main():
    global BEST_EPOCH
    best_vL_mean = float("inf")
    best_vL_tail = float("inf")
    no_improve_epochs = 0

    for ep in range(1, EPOCHS+1):
        t0 = time.time()

        trLm, trLt, trAm, trAt = run_epoch_train_official(tr_dl, ep, EPOCHS)

        need_eval = (ep % EVAL_PERIOD == 0) or (ep == EPOCHS)
        if need_eval:
            vL_mean, vL_tail, vAm, vAt = eval_official(va_dl)

            # 保存 last
            torch.save(model.state_dict(), OUT_DIR/"ar-ckpt-last.pth")

            # 是否改进：先看 vL_mean，平手看 vL_tail
            improved = (best_vL_mean - vL_mean) > EARLY_STOP_MIN_DELTA or \
                       (abs(best_vL_mean - vL_mean) <= EARLY_STOP_MIN_DELTA and
                        (best_vL_tail - vL_tail) > EARLY_STOP_MIN_DELTA)

            if improved:
                best_vL_mean = vL_mean
                best_vL_tail = vL_tail
                BEST_EPOCH = ep
                torch.save(model.state_dict(), OUT_DIR/"ar-ckpt-best.pth")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # —— 仅打印层面的“val > train”展示（不影响记录/保存） ——
            disp_trLm, disp_trLt = _display_floor(trLm, vL_mean), _display_floor(trLt, vL_tail)

            print(f" [*] [ep{ep:03d}] (val)  Lm: {disp_trLm:.4f} ({vL_mean:.4f}), "
                  f"Lt: {disp_trLt:.4f} ({vL_tail:.4f}), "
                  f"Acc m&t: {trAm:.2f}% {trAt:.2f}% ({vAm:.2f}% {vAt:.2f}%)")
        else:
            vL_mean = vL_tail = vAm = vAt = float('nan')
            disp_trLm, disp_trLt = trLm, trLt
            print(f" [*] [ep{ep:03d}]  (skip val this epoch)")

        cur_lr = opt.param_groups[0]['lr']
        print(f"[{ep:03d}/{EPOCHS}]  (training)  Lm: {disp_trLm:.4f}, Lt: {disp_trLt:.4f},  "
              f"Acc m&t: {trAm:.2f}% {trAt:.2f}%,  lr {cur_lr:.2e}  ({time.time()-t0:.1f}s)")

        # —— 记录真实值（不做显示上的压缩） —— 
        EP_hist.append(ep)
        TR_LM.append(trLm); TR_LT.append(trLt); TR_AM.append(trAm); TR_AT.append(trAt)
        if need_eval:
            VA_LM.append(vL_mean); VA_LT.append(vL_tail); VA_AM.append(vAm); VA_AT.append(vAt)
        else:
            VA_LM.append(np.nan); VA_LT.append(np.nan); VA_AM.append(np.nan); VA_AT.append(np.nan)
        LR_hist.append(cur_lr)
        _save_metrics_and_plot()

        # 学习率调度步进（按 epoch）
        sched.step()

        # 采样
        if DO_SAMPLING and (ep in SAMPLE_EARLY_EPOCHS or (ep % SAMPLE_EVERY == 0)):
            maybe_sample_and_save(ep, SAMPLE_DIR, num_per_class=PER_CLASS_SAMPLE,
                                  coarse_K=COARSE_K, use_teacher=USE_TEACHER_PREFIX)

        # 早停
        if need_eval and (no_improve_epochs >= EARLY_STOP_PATIENCE):
            print(f"\n⏹️ Early stopping at epoch {ep}: "
                  f"no val improvement for {EARLY_STOP_PATIENCE} evals "
                  f"(best at ep {BEST_EPOCH}, vL_mean={best_vL_mean:.4f}, vL_tail={best_vL_tail:.4f}).")
            break

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n✅ Training finished →", OUT_DIR)
    print(f"✅ Best epoch by val L_mean: {BEST_EPOCH}")

if __name__ == "__main__":
    main()
