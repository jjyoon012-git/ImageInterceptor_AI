# 1차 분류기 학습 코드. 욜로랑 분류
import os, argparse, time, math, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from convnext_multihead import ConvNeXtTinyMultiHead
from mixed_dataset import make_dls
from loss_utils import masked_bce_with_logits

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_by_head(logits, targets, thr=0.5):
    # 마스크(-1) 제외하고 정확도 리턴 (nudity_acc, violence_acc)
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= thr).long()
        mask_n = (targets[:, 0] >= 0)
        mask_v = (targets[:, 1] >= 0)
        acc_n = (preds[mask_n, 0] == targets[mask_n, 0].long()).float().mean().item() if mask_n.any() else float("nan")
        acc_v = (preds[mask_v, 1] == targets[mask_v, 1].long()).float().mean().item() if mask_v.any() else float("nan")
        return acc_n, acc_v

def train_one_epoch(model, dl, optimizer, scheduler, device, scaler=None, grad_clip=1.0):
    model.train()
    total_loss, n_samples = 0.0, 0
    acc_n_sum, acc_v_sum, acc_n_cnt, acc_v_cnt = 0.0, 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = masked_bce_with_logits(logits, y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = masked_bce_with_logits(logits, y)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

        n_acc, v_acc = accuracy_by_head(logits, y)
        if not math.isnan(n_acc):
            acc_n_sum += n_acc; acc_n_cnt += 1
        if not math.isnan(v_acc):
            acc_v_sum += v_acc; acc_v_cnt += 1

    avg_loss = total_loss / max(n_samples, 1)
    avg_n = (acc_n_sum / acc_n_cnt) if acc_n_cnt else float("nan")
    avg_v = (acc_v_sum / acc_v_cnt) if acc_v_cnt else float("nan")
    return avg_loss, avg_n, avg_v

@torch.no_grad()
def validate(model, dl, device):
    model.eval()
    total_loss, n_samples = 0.0, 0
    acc_n_sum, acc_v_sum, acc_n_cnt, acc_v_cnt = 0.0, 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = masked_bce_with_logits(logits, y)
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

        n_acc, v_acc = accuracy_by_head(logits, y)
        if not math.isnan(n_acc):
            acc_n_sum += n_acc; acc_n_cnt += 1
        if not math.isnan(v_acc):
            acc_v_sum += v_acc; acc_v_cnt += 1

    avg_loss = total_loss / max(n_samples, 1)
    avg_n = (acc_n_sum / acc_n_cnt) if acc_n_cnt else float("nan")
    avg_v = (acc_v_sum / acc_v_cnt) if acc_v_cnt else float("nan")
    return avg_loss, avg_n, avg_v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nud_root", default="data/classification/nudity")
    ap.add_argument("--vio_root", default="data/classification/violence")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="CUDA AMP 사용")
    ap.add_argument("--out", default="runs/convnext_tiny_multihead.pth")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    # Data
    train_dl, val_dl = make_dls(args.nud_root, args.vio_root, bs=args.bs, num_workers=4)

    # Model / Optim / Sched
    model = ConvNeXtTinyMultiHead().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl))
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    best_val = float("inf")
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        tr_loss, tr_n, tr_v = train_one_epoch(model, train_dl, optimizer, scheduler, device, scaler)
        va_loss, va_n, va_v = validate(model, val_dl, device)
        print(f"ep{ep:02d} | train {tr_loss:.4f} (nud {tr_n:.3f} / vio {tr_v:.3f}) "
              f"| val {va_loss:.4f} (nud {va_n:.3f} / vio {va_v:.3f})")

        # 베스트 저장(검증 손실 기준)
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.out)
            print(f"[SAVE] {args.out}  (val_loss={va_loss:.4f})")

    print(f"[DONE] elapsed {(time.time()-t0):.1f}s | best val {best_val:.4f}")

if __name__ == "__main__":
    main()
