# Scripts/eval.py
# 2025-09-11 수정 버전: 그냥 nudity랑 violence 합쳐서 multihead 안 쓰고 Binary Classification 수행
import argparse, json, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from torchvision import datasets, transforms, models

# PIL: 손상 이미지 허용 
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def _is_ok_image(path: str) -> bool:
    """손상/비이미지 파일을 사전 필터링"""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

# 변환
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# 멀티헤드 안 씀!! 수정함
def build_binary_model():
    model = models.convnext_tiny(weights=None)
    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, 2)  # 0/1 이진 분류
    return model

def load_ckpt_to_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    # train_1.py 저장형식: {"model": state_dict, ...}
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = build_binary_model().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

# 데이터 로딩
def _make_loader(root, batch_size=32, workers=0):
    ds = datasets.ImageFolder(root, transform=TF, is_valid_file=_is_ok_image)
    if len(ds) == 0:
        raise ValueError(f"No valid images found in: {root}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    return ds, loader

@torch.no_grad()
def collect_probs_binary(model, root, device, batch_size=32, workers=0):
    """
    return:
        y_true (N,), y_prob1 (N,), paths (N,)
    y_prob1 = P(class==1) = softmax(logits)[:,1]
    """
    ds, loader = _make_loader(root, batch_size=batch_size, workers=workers)
    y_true, y_prob1, paths = [], [], []

    # ImageFolder 인덱스 순서에 맞춘 경로
    sample_paths = [p for p, _ in ds.samples]
    offset = 0

    for x, y in loader:
        bsz = x.size(0)
        x = x.to(device, non_blocking=True)
        logits = model(x)                          # [B,2]
        probs  = torch.softmax(logits, dim=1)[:,1] # P(y=1)
        y_true.extend(y.numpy())
        y_prob1.extend(probs.detach().cpu().numpy())
        paths.extend(sample_paths[offset:offset+bsz])
        offset += bsz

    return np.array(y_true), np.array(y_prob1), paths

# 결과 확인용
def _save_samples(paths, y_true, y_prob, threshold, outdir, tag="test"):
    """
    success: 정답 중 |p-0.5| 큰 상위 3
    fail   : 오답 중 |p-0.5| 큰 상위 3
    """
    pred = (y_prob >= threshold).astype(int)
    conf = np.abs(y_prob - 0.5)

    idx_success = np.where(pred == y_true)[0]
    idx_fail    = np.where(pred != y_true)[0]

    idx_success = idx_success[np.argsort(-conf[idx_success])]
    idx_fail    = idx_fail[np.argsort(-conf[idx_fail])]

    pick_s = idx_success[:3]
    pick_f = idx_fail[:3]

    dst_s = Path(outdir) / "samples" / tag / "success"
    dst_f = Path(outdir) / "samples" / tag / "fail"
    dst_s.mkdir(parents=True, exist_ok=True) 
    dst_f.mkdir(parents=True, exist_ok=True)  

    def _tag(p, y, pr):
        stem = Path(p).stem
        ext  = Path(p).suffix.lower()
        return f"{stem}_y{int(y)}_p{pr:.3f}{ext}"

    for i in pick_s:
        src = paths[i]; y = y_true[i]; pr = y_prob[i]
        try:
            shutil.copy2(src, dst_s / _tag(src, y, pr))
        except Exception:
            pass

    for i in pick_f:
        src = paths[i]; y = y_true[i]; pr = y_prob[i]
        try:
            shutil.copy2(src, dst_f / _tag(src, y, pr))
        except Exception:
            pass

# 시각화
def plot_confusion(y_true, y_pred, title, outpath):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0","1"])
    disp.plot(cmap=plt.cm.Blues, values_format="d", colorbar=False)
    plt.title(title); plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

def plot_roc(y_true, y_prob, title, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {title}"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def plot_pr(y_true, y_prob, title, outpath):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve - {title}"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

# 경로 수정하세요
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val",  default=r"C:\Users\jjeong\Desktop\ImageInterceptor\data\Classification\merge\val")
    ap.add_argument("--test", default=r"C:\Users\jjeong\Desktop\ImageInterceptor\data\Classification\merge\test")
    ap.add_argument("--ckpt", default=r"C:\Users\jjeong\Desktop\ImageInterceptor\checkpoints\best.pt")
    ap.add_argument("--outdir", default=r"C:\Users\jjeong\Desktop\ImageInterceptor\runs\eval_binary")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)   # Windows 안전값
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_ckpt_to_model(args.ckpt, device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}

    #  VAL 
    yv, pv, pathsv = collect_probs_binary(
        model, args.val, device, batch_size=args.batch_size, workers=args.workers
    )
    pred_v = (pv >= args.threshold).astype(int)
    fpr_v, tpr_v, _ = roc_curve(yv, pv)
    results["val"] = {
        "auroc": float(auc(fpr_v, tpr_v)),
        "auprc": float(average_precision_score(yv, pv)),
        "threshold": args.threshold,
        "n": int(len(yv)),
        "acc": float(((pred_v == yv).sum()) / len(yv))
    }
    plot_confusion(yv, pred_v, "Val Confusion", outdir / "cm_val.png")
    plot_roc(yv, pv, "Val", outdir / "roc_val.png")
    plot_pr(yv, pv, "Val", outdir / "pr_val.png")
    _save_samples(pathsv, yv, pv, args.threshold, outdir, "val")

    # TEST 
    yt, pt, pathst = collect_probs_binary(
        model, args.test, device, batch_size=args.batch_size, workers=args.workers
    )
    pred_t = (pt >= args.threshold).astype(int)
    fpr_t, tpr_t, _ = roc_curve(yt, pt)
    results["test"] = {
        "auroc": float(auc(fpr_t, tpr_t)),
        "auprc": float(average_precision_score(yt, pt)),
        "threshold": args.threshold,
        "n": int(len(yt)),
        "acc": float(((pred_t == yt).sum()) / len(yt))
    }
    plot_confusion(yt, pred_t, "Test Confusion", outdir / "cm_test.png")
    plot_roc(yt, pt, "Test", outdir / "roc_test.png")
    plot_pr(yt, pt, "Test", outdir / "pr_test.png")
    _save_samples(pathst, yt, pt, args.threshold, outdir, "test")

    # Save metrics
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[DONE] metrics saved:", results)
    print(f"[SAMPLES] {outdir / 'samples'}")

if __name__ == "__main__":
    main()
