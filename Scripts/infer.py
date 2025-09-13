# 2025-09-13 재업로드
import argparse, json
from pathlib import Path
from typing import Dict, Any, Iterable

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile, UnidentifiedImageError

# 손상 이미지 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 사용자 맞춤 필터링
# 1(약)
# 2(중)
# 3(강)
FILTER_LEVEL_THRESHOLDS = {
    1: 0.40,  
    2: 0.50,  
    3: 0.65, # 임계치 첫 번째 버전. 확인 후 수정 필요
}

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_model(ckpt_path: str, device: str) -> nn.Module:
    model = models.convnext_tiny(weights=None)
    in_f = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_f, 2)  # binary (0/1)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # DataParallel 호환
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

def _load_image_safe(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.verify()
    return Image.open(path).convert("RGB")

@torch.no_grad()
def infer_one(model: nn.Module, path: Path, device: str, threshold: float) -> Dict[str, Any]:
    try:
        img = _load_image_safe(path)
        x = TF(img).unsqueeze(0).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)                 # [1,2]
            prob1 = torch.softmax(logits, dim=1)[0,1].item()  # P(y=1 harmful)
        label = "harmful" if prob1 >= threshold else "safe"
        return {"path": str(path), "label": label, "p_harmful": float(prob1), "threshold": float(threshold)}
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return {"path": str(path), "error": f"{type(e).__name__}: {e}"}

def _iter_images(folder: Path, recursive: bool, exts: Iterable[str]):
    globber = folder.rglob if recursive else folder.glob
    for p in globber("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def resolve_threshold(level: int | None, override: float | None) -> float:
    if override is not None:
        return float(override)
    if level is None:
        level = 2
    if level not in FILTER_LEVEL_THRESHOLDS:
        raise SystemExit(f"--level 은 1|2|3 중 하나여야 합니다 (입력: {level})")
    return FILTER_LEVEL_THRESHOLDS[level]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="단일 이미지 경로")
    ap.add_argument("--folder", help="폴더 경로 (일괄 추론)")
    ap.add_argument("--recursive", action="store_true", help="폴더 재귀 탐색")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.webp", help="확장자 CSV")
    ap.add_argument("--ckpt", default=r"C:\Users\jjeong\Desktop\ImageInterceptor\checkpoints\best.pt")
    ap.add_argument("--out",  default=r"C:\Users\jjeong\Desktop\ImageInterceptor\runs\infer_results.json")
    # 필터 강도 및 임계치 직접 지정할 수도 있음
    ap.add_argument("--level", type=int, choices=[1,2,3], help="필터 강도(1=약, 2=중, 3=강)")
    ap.add_argument("--threshold", type=float, help="임계값을 직접 지정 (레벨보다 우선)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    # threshold
    threshold = resolve_threshold(args.level, args.threshold)

    exts = {e.strip().lower() if e.strip().startswith(".") else "."+e.strip().lower()
            for e in args.exts.split(",") if e.strip()}

    results = []
    if args.image:
        results.append(infer_one(model, Path(args.image), device, threshold))
    elif args.folder:
        for p in _iter_images(Path(args.folder), args.recursive, exts):
            results.append(infer_one(model, p, device, threshold))
    else:
        raise SystemExit(" --image 또는 --folder 중 하나는 지정해야 합니다.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 저장: {args.out}  (총 {len(results)}개, level={args.level or 2}, threshold={threshold:.2f})")
    for r in results[:5]:
        print(r)

if __name__ == "__main__":
    main()
