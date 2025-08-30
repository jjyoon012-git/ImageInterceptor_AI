from pathlib import Path
from typing import List, Tuple
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ====== 기본 설정 ======
IMG_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 학습/검증 변환 (ConvNeXt-Tiny 권장 전처리)
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _list_images(root: Path) -> List[Path]:
    """root/{0,1}/* 에서 이미지 경로만 수집"""
    if not root.exists():
        return []
    out: List[Path] = []
    for cls in ("0", "1"):
        d = root / cls
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                out.append(p)
    return out


class MixedDataset(Dataset):
    """
    서로 다른 루트의 이진 데이터셋(nudity, violence)을 합쳐
    라벨을 [nudity, violence]로 구성.
      - nudity 샘플: [0/1, -1]
      - violence 샘플: [-1, 0/1]
    -1 라벨은 학습 손실에서 마스킹 (ignore) 처리하세요.
    """
    def __init__(self,
                 nudity_root: str,
                 violence_root: str,
                 split: str = "train",
                 transform=None):
        self.transform = transform

        nud_paths = _list_images(Path(nudity_root) / split)
        vio_paths = _list_images(Path(violence_root) / split)

        self.samples: List[Tuple[Path, torch.Tensor]] = []

        # nudity: [y, -1]
        for p in nud_paths:
            y = 1 if p.parent.name == "1" else 0
            self.samples.append((p, torch.tensor([y, -1], dtype=torch.float32)))

        # violence: [-1, y]
        for p in vio_paths:
            y = 1 if p.parent.name == "1" else 0
            self.samples.append((p, torch.tensor([-1, y], dtype=torch.float32)))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under:\n  {Path(nudity_root)/split}\n  {Path(violence_root)/split}\n"
                "Expected structure: root/{train,val,test}/{0,1}/image.jpg"
            )

    def __len__(self):
        return len(self.samples)

    def _safe_load(self, path: Path):
        # 손상/비이미지 대비 안전 로드
        try:
            img = Image.open(path).convert("RGB")
            return img
        except (UnidentifiedImageError, OSError):
            return None

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = self._safe_load(path)
        # 만약 깨진 파일이면 다음 샘플로 대체
        if img is None:
            # 순환적으로 다음 항목 시도
            j = (idx + 1) % len(self.samples)
            while j != idx:
                img = self._safe_load(self.samples[j][0])
                if img is not None:
                    path, y = self.samples[j]
                    break
                j = (j + 1) % len(self.samples)
            if img is None:
                # 전부 실패 시 에러
                raise RuntimeError(f"Failed to read any image around index {idx}.")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, y


def make_dls(nud_root: str,
             vio_root: str,
             bs: int = 32,
             num_workers: int = 4):
    # 학습/검증 DataLoader 생성
    train_ds = MixedDataset(nud_root, vio_root, split="train", transform=train_tf)
    val_ds   = MixedDataset(nud_root, vio_root, split="val",   transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


# 선택: 빠른 카운트 유틸
def quick_count(nud_root: str, vio_root: str):
    for root in (nud_root, vio_root):
        root = Path(root)
        for split in ("train", "val", "test"):
            for cls in ("0", "1"):
                d = root / split / cls
                n = len(_list_images(d.parent)) if d.exists() else 0
            # 위 한 줄은 split 폴더 전체 개수 합산용으로 간단화됨
        # 필요하면 세부 프린트를 원하실 때 아래처럼 사용:
        # print(f"{root}/{split}/0: {len(_list_images(root/split))} ...")
