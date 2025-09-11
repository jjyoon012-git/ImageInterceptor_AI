# train_1.py
# Binary classification (0: 비유해, 1: 유해[nudity+violence])
# Folder layout (already split 3:1:1):
#   merge/
#     train/{0,1}
#     val/{0,1}
#     test/{0,1}

import os, time, math
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 일부 손상/잘린 이미지 허용


# -------------------------
# 고정 경로 & 하이퍼파라미터 (원하면 여기만 수정)
# -------------------------
DATA_ROOT = r"C:\Users\jjeong\Desktop\ImageInterceptor\data\Classification\merge"
OUT_DIR   = r"C:\Users\jjeong\Desktop\ImageInterceptor\checkpoints"

IMG_SIZE  = 224
BATCH     = 32
EPOCHS    = 30
LR        = 3e-4
WORKERS   = 4           # 윈도우에서 문제나면 0으로
PATIENCE  = 5
FREEZE_BACKBONE = False # True면 분류기만 학습
SEED = 42
USE_CLASS_WEIGHTS = True  # 학습용 가중치로 불균형 완화
USE_SAMPLER = False        # 필요시 WeightedRandomSampler로 대체 가능


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def safe_pil_loader(path: str):
    # 파일을 바이너리로 열고 즉시 RGB로 변환(지연 로딩 중 오류 감소)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def build_dataloaders(root: Path, img_size: int, batch_size: int, num_workers: int):
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), normalize
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), normalize
    ])

    train_ds = datasets.ImageFolder(str(root/"train"), transform=train_tf, loader=safe_pil_loader)
    val_ds   = datasets.ImageFolder(str(root/"val"),   transform=eval_tf,  loader=safe_pil_loader)
    test_ds  = datasets.ImageFolder(str(root/"test"),  transform=eval_tf,  loader=safe_pil_loader)

    common = dict(batch_size=batch_size,
                  num_workers=num_workers,
                  pin_memory=True,
                  persistent_workers=(num_workers>0))

    if USE_SAMPLER:
        # 클래스 불균형이 심하면 샘플러 사용 (간단 버전: class count 역수)
        import numpy as np
        targets = np.array(train_ds.targets)
        counts = np.bincount(targets)
        weights = 1.0 / (counts[targets] + 1e-6)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **common)

    val_loader  = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, train_ds

def build_model(num_classes=2, lr=3e-4, freeze=False):
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)
    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, num_classes)

    if freeze:
        for n, p in model.named_parameters():
            if not n.startswith("classifier."):
                p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    return model, optimizer, scheduler

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    tp = fp = fn = tn = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = ce(logits, y)

        total += loss.item() * x.size(0)
        n += x.size(0)

        preds = torch.argmax(logits, dim=1)
        tp += ((preds==1) & (y==1)).sum().item()
        tn += ((preds==0) & (y==0)).sum().item()
        fp += ((preds==1) & (y==0)).sum().item()
        fn += ((preds==0) & (y==1)).sum().item()

    avg_loss = total / max(n,1)
    acc = (tp + tn) / max((tp+tn+fp+fn), 1)
    prec = tp / max((tp+fp), 1)
    rec  = tp / max((tp+fn), 1)
    f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return avg_loss, acc, prec, rec, f1

def main():
    set_seed(SEED)
    root = Path(DATA_ROOT)
    assert (root/"train").exists() and (root/"val").exists() and (root/"test").exists(), \
        f"폴더 구조 확인: {root}/(train|val|test)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir/"best.pt"

    train_loader, val_loader, test_loader, train_ds = build_dataloaders(root, IMG_SIZE, BATCH, WORKERS)

    # 클래스 가중치(선택)
    if USE_CLASS_WEIGHTS:
        import numpy as np
        counts = np.bincount(np.array(train_ds.targets))
        # inverse frequency를 정규화해서 사용
        class_weights = counts.sum() / (2.0*counts + 1e-6)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    model, optimizer, scheduler = build_model(num_classes=2, lr=LR, freeze=FREEZE_BACKBONE)
    model.to(device)

    # 최신 AMP API로 변경
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

    best_val = math.inf
    bad = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        scheduler.step()
        train_loss = run_loss / max(seen,1)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)

        print(f"[{epoch:03d}] train {train_loss:.4f} | "
              f"val {val_loss:.4f} acc {val_acc:.4f} prec {val_prec:.4f} rec {val_rec:.4f} f1 {val_f1:.4f} "
              f"({time.time()-t0:.1f}s)")

        # 체크포인트 & 조기 종료
        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "img_size": IMG_SIZE,
                "model_name": "convnext_tiny",
                "classes": train_ds.classes
            }, ckpt_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    # Best ckpt 로드 후 테스트
    sd = torch.load(ckpt_path, map_location="cpu")["model"]
    model.load_state_dict(sd)
    model.to(device)
    te_loss, te_acc, te_prec, te_rec, te_f1 = evaluate(model, test_loader, device)
    print(f"[TEST] loss {te_loss:.4f} acc {te_acc:.4f} prec {te_prec:.4f} rec {te_rec:.4f} f1 {te_f1:.4f}")
    print(f"Saved checkpoint: {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
