# 2025년 9월 5일 업데이트
# CNN 기반 모델의 추론 스크립트입니다.
# 3단계(약/중/강) 필터링 강도를 --level {1,2,3}로 선택할 수 있습니다.
# Nudity 우선 정책은 동일하게 적용됩니다.
# 현재 카테고리는 Nudity, Violence 두 가지로 구성되어 있습니다.

import argparse, json
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from convnext_multihead import ConvNeXtTinyMultiHead

# 디폴트 임계값 설정 
# TN: True Nudity, TV: True Violence
DEFAULT_TN_LOW, DEFAULT_TN_HIGH = 0.30, 0.60 # 일단 초기 세팅
DEFAULT_TV_LOW, DEFAULT_TV_HIGH = 0.30, 0.60 # 일단 초기 세팅

# 1. 약함 / 2. 중간 / 3. 강함 임계값 설정 => 이거로 필터링 단계 조절

LEVEL_PRESETS = {
    1: dict(tn_low=0.35, tn_high=0.80, tv_low=0.30, tv_high=0.75),  # 약한
    2: dict(tn_low=0.25, tn_high=0.65, tv_low=0.20, tv_high=0.60),  # 중간
    3: dict(tn_low=0.15, tn_high=0.50, tv_low=0.10, tv_high=0.45),  # 강한
}

#확률값(pn=누드, pv=폭력)으로 최종 라벨 결정 (Nudity 우선 정책)
def decide(pn, pv, tn_low, tn_high, tv_low, tv_high):
    if pn >= tn_high:
        return "harmful_nudity", pn
    if pn <= tn_low:
        if pv >= tv_high: return "harmful_violence", pv
        if pv <= tv_low:  return "safe", max(pn, pv)
        return "abstain_violence", pv
    return "abstain_nudity", pn

# 이미지 전처리. 따라서 서버, 확장프로그램은 신경 안 써도 되는 내용입니다.
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

def infer_one(model, path, device, tn_low, tn_high, tv_low, tv_high):
    img = Image.open(path).convert("RGB")
    x = TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pn, pv = torch.sigmoid(logits)[0].tolist()
    label, score = decide(pn, pv, tn_low, tn_high, tv_low, tv_high)
    return {"path": str(path), "label": label, "pn": pn, "pv": pv, "score": score}

#  --level 프리셋을 반영하되, 개별 임계값 인자가 주어지면 그 값으로 오버라이드
def resolve_thresholds(args):
    preset = LEVEL_PRESETS.get(args.level, LEVEL_PRESETS[2])
    tn_low  = args.tn_low  if args.tn_low  is not None else preset["tn_low"]
    tn_high = args.tn_high if args.tn_high is not None else preset["tn_high"]
    tv_low  = args.tv_low  if args.tv_low  is not None else preset["tv_low"]
    tv_high = args.tv_high if args.tv_high is not None else preset["tv_high"]
    return tn_low, tn_high, tv_low, tv_high


# main 함수.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="단일 이미지 경로")
    ap.add_argument("--folder", help="폴더 경로 (일괄 추론)")
    ap.add_argument("--ckpt", default="runs/convnext_tiny_multihead.pth")
    ap.add_argument("--out", default="runs/infer_results.json")

    # 필터 강도 선택(1=약, 2=중, 3=강)
    ap.add_argument("--level", type=int, choices=[1,2,3], default=2,
                    help="필터링 강도 선택: 1=약함, 2=중간(기본값), 3=강함")

    # (선택) 개별 임계값 오버라이드 옵션 => GPT가 제안함.
    ap.add_argument("--tn_low", type=float, default=None)
    ap.add_argument("--tn_high", type=float, default=None)
    ap.add_argument("--tv_low", type=float, default=None)
    ap.add_argument("--tv_high", type=float, default=None)

    args = ap.parse_args()

    if not (args.image or args.folder):
        raise SystemExit(" --image 또는 --folder 중 하나는 지정해야 합니다.")

    # 단계별 프리셋 + 오버라이드 반영
    tn_low, tn_high, tv_low, tv_high = resolve_thresholds(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    results = []
    if args.image:
        results.append(infer_one(model, Path(args.image), device, tn_low, tn_high, tv_low, tv_high))
    elif args.folder:
        exts = {".jpg",".jpeg",".png",".bmp",".webp"}
        for p in Path(args.folder).glob("*"):
            if p.suffix.lower() in exts:
                results.append(infer_one(model, p, device, tn_low, tn_high, tv_low, tv_high))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2) # json파일로 저장

    print(f"[DONE] 결과 {args.out} 저장 완료")
    for r in results[:5]:  # 앞의 몇 개만 미리보기
        print(r)
    print(f"[LEVEL={args.level}] tn_low={tn_low}, tn_high={tn_high}, tv_low={tv_low}, tv_high={tv_high}")  # 로그 확인용

if __name__ == "__main__":
    main()
