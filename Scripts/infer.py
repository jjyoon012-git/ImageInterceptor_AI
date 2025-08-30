import argparse, json
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from convnext_multihead import ConvNeXtTinyMultiHead

# === 임계값 (Nudity 우선 정책) ===
Tn_low, Tn_high = 0.25, 0.65
Tv_low, Tv_high = 0.20, 0.60

def decide(pn, pv):
    """확률값(pn=누드, pv=폭력)으로 최종 라벨 결정"""
    if pn >= Tn_high:
        return "harmful_nudity", pn
    if pn <= Tn_low:
        if pv >= Tv_high: return "harmful_violence", pv
        if pv <= Tv_low:  return "safe", max(pn, pv)
        return "abstain_violence", pv
    return "abstain_nudity", pn

# === 전처리 ===
TF = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_model(ckpt, device):
    model = ConvNeXtTinyMultiHead().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model

def infer_one(model, path, device):
    img = Image.open(path).convert("RGB")
    x = TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pn, pv = torch.sigmoid(logits)[0].tolist()
    label, score = decide(pn, pv)
    return {"path": str(path), "label": label, "pn": pn, "pv": pv, "score": score}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="단일 이미지 경로")
    ap.add_argument("--folder", help="폴더 경로 (일괄 추론)")
    ap.add_argument("--ckpt", default="runs/convnext_tiny_multihead.pth")
    ap.add_argument("--out", default="runs/infer_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    results = []
    if args.image:
        results.append(infer_one(model, Path(args.image), device))
    elif args.folder:
        exts = {".jpg",".jpeg",".png",".bmp",".webp"}
        for p in Path(args.folder).glob("*"):
            if p.suffix.lower() in exts:
                results.append(infer_one(model, p, device))
    else:
        raise SystemExit(" --image 또는 --folder 중 하나는 지정해야 합니다.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 결과 {args.out} 저장 완료")
    for r in results[:5]:  # 앞의 몇 개만 미리보기
        print(r)

if __name__ == "__main__":
    main()
