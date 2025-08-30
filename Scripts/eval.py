# Scripts/eval.py
import argparse, json, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless 저장
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from torchvision import datasets, transforms

from convnext_multihead import ConvNeXtTinyMultiHead

# ---------- PIL: 손상 이미지 허용 + 사전 검증 ----------
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # 초대형 경고 해제(선택)

def _is_ok_image(path: str) -> bool:
    """ImageFolder 인덱싱 단계에서 손상/비이미지 파일 제외"""
    try:
        with Image.open(path) as im:
            im.verify()  # 헤더 무결성
        with Image.open(path) as im:
            im.convert("RGB").load()  # 실제 디코딩
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

# ---------- Transforms ----------
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ---------- Model ----------
def load_model(ckpt, device):
    model = ConvNeXtTinyMultiHead().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ---------- Data ----------
def _make_loader(root, batch_size=32, workers=0):
    ds = datasets.ImageFolder(root, transform=TF, is_valid_file=_is_ok_image)
    if len(ds) == 0:
        raise ValueError(f"No valid images found in: {root}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    return ds, loader

def collect_probs(model, root, device, domain, batch_size=32, workers=0):
    """
    return: y_true (N,), y_prob (N,), paths (N,)
    domain: 'nudity' | 'violence'
    """
    ds, loader = _make_loader(root, batch_size=batch_size, workers=workers)
    y_true, y_prob, paths = [], [], []

    # ImageFolder 순서에 맞춰 경로 추출
    sample_paths = [p for p,_ in ds.samples]
    offset = 0

    for x, y in loader:
        bsz = x.size(0)
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(x)              # [B,2] -> (nudity, violence)
            probs  = torch.sigmoid(logits) # [B,2]
        p = probs[:, 0] if domain == "nudity" else probs[:, 1]

        y_true.extend(y.cpu().numpy())
        y_prob.extend(p.cpu().numpy())
        paths.extend(sample_paths[offset:offset+bsz])
        offset += bsz

    return np.array(y_true), np.array(y_prob), paths

# ---------- Save top samples ----------
def _save_samples(paths, y_true, y_prob, threshold, outdir, domain):
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

    dst_s = Path(outdir) / "samples" / domain / "success"
    dst_f = Path(outdir) / "samples" / domain / "fail"
    dst_s.mkdir(parents=True, exist_ok=True)
    dst_f.mkdir(parents=True, exist_ok=True)

    def _tag(p, y, pr):
        stem = Path(p).stem
        ext  = Path(p).suffix.lower()
        return f"{stem}_y{int(y)}_p{pr:.3f}{ext}"

    for i in pick_s:
        src = paths[i]; y = y_true[i]; pr = y_prob[i]
        try: shutil.copy2(src, dst_s / _tag(src, y, pr))
        except Exception: pass

    for i in pick_f:
        src = paths[i]; y = y_true[i]; pr = y_prob[i]
        try: shutil.copy2(src, dst_f / _tag(src, y, pr))
        except Exception: pass

# ---------- Plots ----------
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

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nud_val", default="data/classification/nudity/val")
    ap.add_argument("--vio_val", default="data/classification/violence/val")
    ap.add_argument("--ckpt", default="runs/convnext_tiny_multihead.pth")
    ap.add_argument("--outdir", default="runs/eval")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)   # Windows 안전 값
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ----- Nudity -----
    y_n, p_n, paths_n = collect_probs(
        model, args.nud_val, device, "nudity",
        batch_size=args.batch_size, workers=args.workers
    )
    pred_n = (p_n >= args.threshold).astype(int)
    fpr_n, tpr_n, _ = roc_curve(y_n, p_n)
    results["nudity"] = {
        "auroc": float(auc(fpr_n, tpr_n)),
        "auprc": float(average_precision_score(y_n, p_n)),
        "threshold": args.threshold
    }
    plot_confusion(y_n, pred_n, "Nudity Confusion", outdir/"cm_nudity.png")
    plot_roc(y_n, p_n, "Nudity", outdir/"roc_nudity.png")
    plot_pr(y_n, p_n, "Nudity", outdir/"pr_nudity.png")
    _save_samples(paths_n, y_n, p_n, args.threshold, outdir, "nudity")

    # ----- Violence -----
    y_v, p_v, paths_v = collect_probs(
        model, args.vio_val, device, "violence",
        batch_size=args.batch_size, workers=args.workers
    )
    pred_v = (p_v >= args.threshold).astype(int)
    fpr_v, tpr_v, _ = roc_curve(y_v, p_v)
    results["violence"] = {
        "auroc": float(auc(fpr_v, tpr_v)),
        "auprc": float(average_precision_score(y_v, p_v)),
        "threshold": args.threshold
    }
    plot_confusion(y_v, pred_v, "Violence Confusion", outdir/"cm_violence.png")
    plot_roc(y_v, p_v, "Violence", outdir/"roc_violence.png")
    plot_pr(y_v, p_v, "Violence", outdir/"pr_violence.png")
    _save_samples(paths_v, y_v, p_v, args.threshold, outdir, "violence")

    # ----- Save metrics -----
    with open(outdir/"metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[DONE] metrics saved:", results)
    print(f"[SAMPLES] {outdir / 'samples'}")

if __name__ == "__main__":
    main()
