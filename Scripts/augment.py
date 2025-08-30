import argparse
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import albumentations as A
import numpy as np
import hashlib
import time

# -------- 도우미: 버전 차이를 자동 흡수하는 안전 생성기 --------
def make_rrc(scale, ratio):
    """RandomResizedCrop: v2(size=...), v1(height,width=...) 자동 선택"""
    try:
        # v2
        return A.RandomResizedCrop(size=(256, 256), scale=scale, ratio=ratio, p=1.0)
    except Exception:
        # v1
        return A.RandomResizedCrop(height=256, width=256, scale=scale, ratio=ratio, p=1.0)

def make_geom(rotate):
    """Affine(=v2) 또는 ShiftScaleRotate(=v1) 자동 선택"""
    # 먼저 v2 Affine 시도 (mode 인자 없이!)
    try:
        return A.Affine(
            translate_percent=(0.0, 0.06),
            scale=(0.9, 1.1),
            rotate=(-rotate, rotate),
            p=0.6
        )
    except Exception:
        # v1
        return A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.1,
            rotate_limit=rotate,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.6
        )

def make_noise(var_limit):
    """GaussianNoise(v2) 또는 GaussNoise(v1) 자동 선택"""
    if hasattr(A, "GaussianNoise"):
        return A.GaussianNoise(var_limit=var_limit, p=1.0)
    else:
        return A.GaussNoise(var_limit=var_limit, p=1.0)

def make_compression(jpeg_lo, jpeg_hi):
    """ImageCompression(v2, quality_range) 또는 v1(JpegCompression/ImageCompression) 자동"""
    # v2 ImageCompression(quality_range=(lo,hi))
    try:
        return A.ImageCompression(quality_range=(jpeg_lo, jpeg_hi), p=0.7)
    except Exception:
        # v1 path
        try:
            return A.OneOf([
                A.JpegCompression(quality_lower=jpeg_lo, quality_upper=jpeg_hi, p=0.7),
                A.ImageCompression(quality_lower=jpeg_lo, quality_upper=jpeg_hi, p=0.3),
            ], p=0.4)
        except Exception:
            # 아주 구버전 대비 안전 fallback
            return A.ImageCompression(quality_lower=jpeg_lo, quality_upper=jpeg_hi, p=0.7)

def make_defocus_or_blur():
    """Defocus가 있으면 사용, 없으면 GaussianBlur로 대체"""
    # v2 Defocus(radius=...), v1에는 없을 수 있음
    try:
        return A.Defocus(radius=(2, 4))
    except Exception:
        return A.GaussianBlur(blur_limit=(3, 5))

# --- 안전한(내용 보존형) 증강 파이프라인 (v1/v2 호환) ---
def make_pipeline(strength="medium"):
    """
    strength: 'light' | 'medium' | 'strong'
    """
    s = strength if strength in {"light", "medium", "strong"} else "medium"
    cfg = {
        "light":  dict(crop_scale=(0.85, 1.0), rotate=5,  noise=(3,10),  jpeg=(60,95), blur_p=0.2, color_jitter=0.15),
        "medium": dict(crop_scale=(0.75, 1.0), rotate=10, noise=(5,20), jpeg=(40,90), blur_p=0.35, color_jitter=0.2),
        "strong": dict(crop_scale=(0.65, 1.0), rotate=12, noise=(5,25), jpeg=(30,85), blur_p=0.5,  color_jitter=0.25),
    }[s]

    rrc = make_rrc(cfg["crop_scale"], (0.8, 1.25))
    geom = make_geom(cfg["rotate"])
    noise = make_noise(cfg["noise"])
    comp = make_compression(cfg["jpeg"][0], cfg["jpeg"][1])
    defocus = make_defocus_or_blur()

    return A.Compose([
        rrc,
        A.HorizontalFlip(p=0.5),
        geom,
        A.Perspective(scale=(0.02, 0.05), p=0.2),

        A.OneOf([
            noise,  # Gaussian/Gauss Noise
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.4), p=1.0),
        ], p=0.5),

        comp if isinstance(comp, A.BasicTransform) else comp,  # 그대로 포함

        A.ColorJitter(brightness=cfg["color_jitter"],
                      contrast=cfg["color_jitter"],
                      saturation=cfg["color_jitter"],
                      hue=0.02, p=0.5),

        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5)),
            A.GaussianBlur(blur_limit=(3, 5)),
            defocus,
        ], p=cfg["blur_p"]),

        A.CoarseDropout(max_holes=1, max_height=0.10, max_width=0.10, p=0.15),
    ])

def list_images(d: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in d.glob("*") if p.suffix.lower() in exts])

def save_image(dst_path: Path, img_bgr: np.ndarray):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"Failed to write: {dst_path}")

def unique_name(src: Path, idx: int):
    h = hashlib.md5(f"{src.stem}-{time.time_ns()}-{idx}".encode()).hexdigest()[:8]
    return f"{src.stem}_aug_{h}.jpg"

def augment_split(split_dir: Path, multiplier: float, target: int, strength: str, out_inplace: bool):
    """
    split_dir: e.g., data/classification/violence/train
    multiplier: 각 원본 0 이미지당 생성 배수(정수/실수 가능; 실수면 기대치로 처리)
    target: 라벨 0 최종 목표 수(>0이면 우선). multiplier와 함께 쓰면 target이 우선.
    out_inplace: True면 같은 폴더(train/0)에 저장, False면 train/0_aug에 저장
    """
    zero_dir = split_dir / "0"
    if not zero_dir.exists():
        print(f"[WARN] Skip: {zero_dir} not found.")
        return

    imgs = list_images(zero_dir)
    if len(imgs) == 0:
        print(f"[WARN] No images in {zero_dir}")
        return

    out_dir = zero_dir if out_inplace else split_dir / "0_aug"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 현재 보유량
    current = len(list_images(zero_dir)) + (0 if out_inplace else len(list_images(out_dir)))

    # 생성 개수 계산
    if target and target > current:
        total_to_make = target - current
    else:
        add_per_image = max(multiplier - 1.0, 0.0)
        total_to_make = int(round(len(imgs) * add_per_image))

    if total_to_make <= 0:
        print(f"[INFO] Nothing to augment for {split_dir}. current={current}, target={target}, multiplier={multiplier}")
        return

    pipe = make_pipeline(strength)
    print(f"[INFO] {split_dir} -> generating {total_to_make} augmented images into {out_dir}")

    made = 0
    rng = random.Random(42)
    while made < total_to_make:
        src = Path(imgs[rng.randrange(len(imgs))])
        img = cv2.imread(str(src))
        if img is None:
            print(f"[WARN] Failed to read {src}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = pipe(image=img)["image"]
        aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

        out_name = unique_name(src, made)
        save_image(out_dir / out_name, aug_bgr)
        made += 1

    # 요약
    final_count = len(list_images(zero_dir)) + (0 if out_inplace else len(list_images(out_dir)))
    print(f"[DONE] {split_dir}: generated {made} images. Final (approx) count: {final_count}")

def main():
    parser = argparse.ArgumentParser(description="Augment only label-0 (non-violent) images in violence dataset.")
    parser.add_argument("--root", type=str, default="data/classification/violence",
                        help="violence 카테고리 루트 (예: data/classification/violence)")
    parser.add_argument("--splits", type=str, default="train",
                        help="대상 스플릿(쉼표 구분): train 또는 train,val,test")
    parser.add_argument("--multiplier", type=float, default=2.0,
                        help="배수 증강 (예: 2.0 => 원본당 1장 추가)")
    parser.add_argument("--target", type=int, default=0,
                        help="라벨0 최종 목표 수(우선 적용). 0이면 multiplier 사용")
    parser.add_argument("--strength", type=str, default="medium",
        choices=["light", "medium", "strong"], help="증강 강도")
    parser.add_argument("--inplace", action="store_true",
                        help="같은 폴더(0)에 저장. 지정 안 하면 0_aug 폴더에 저장")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")

    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_dir = root / sp
        if not split_dir.exists():
            print(f"[WARN] Split '{sp}' not found under {root}, skip.")
            continue
        augment_split(split_dir, multiplier=args.multiplier,
                      target=args.target, strength=args.strength,
                      out_inplace=args.inplace)

if __name__ == "__main__":
    main()

