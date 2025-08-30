import shutil, yaml, re
from pathlib import Path
from tqdm import tqdm

# 원본 export 경로
SRC_DIRS = {
    "Nudity/Sexual": Path("data/nudity"),
    "Violence/Gore": Path("data/violence"),
    "Self-harm/Suicide": Path("data/selfharm"),
    "Weapons/Crime": Path("data/weapon"),
}

# 최종 출력 경로
DST = Path("data/merged")
for p in [DST/"images/train", DST/"images/val", DST/"labels/train", DST/"labels/val"]:
    p.mkdir(parents=True, exist_ok=True)

# 대분류 클래스 고정(인덱스 순서)
FINAL_CLASSES = ["nudity", "violence", "selfharm", "weapon"]
FINAL_INDEX = {k:i for i,k in enumerate(FINAL_CLASSES)}

# 라벨명 키워드 → 대분류 매핑(라벨명이 정확히 뭔지 몰라도 키워드로 분류)
KEYWORDS = {
    "nudity":    [r"nude", r"nudity", r"adult", r"breast", r"genital", r"buttock", r"porn"],
    "violence":  [r"violence", r"violent", r"fight", r"punch", r"blood", r"gore", r"explosion"],
    "selfharm":  [r"self[- ]?harm", r"suicid", r"cut", r"razor", r"wrist", r"hang"],
    "weapon":   [r"weapon", r"gun", r"knife", r"rifle", r"pistol", r"blade", r"bomb"],
}

def guess_super(label_name: str) -> str | None:
    ln = label_name.lower()
    for superk, patterns in KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, ln):
                return superk
    return None

def parse_yaml_names(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        # YOLOv5 스타일 {0:name0, 1:name1,...}
        idx2name = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    else:
        idx2name = names
    return idx2name

def convert_split(src_root: Path, split_name_in: str, split_name_out: str):
    # src_root/{train/val or train/valid}/(images|labels)
    img_dir = src_root / split_name_in / "images"
    lbl_dir = src_root / split_name_in / "labels"
    if not img_dir.exists():
        return

    yaml_candidates = list(src_root.glob("*.yaml")) + list((src_root/"..").glob("*.yaml"))
    # 라벨명 불러오기
    idx2name = None
    for yml in yaml_candidates:
        try:
            idx2name = parse_yaml_names(yml)
            if idx2name: break
        except Exception:
            pass
    if idx2name is None:
        print(f"[WARN] names not found for {src_root}")
        idx2name = []

    for img_path in tqdm(list(img_dir.glob("*.*")), desc=f"{src_root.name}:{split_name_out}"):
        # 이미지 복사
        new_img = DST/f"images/{split_name_out}/{img_path.name}"
        shutil.copy2(img_path, new_img)

        # 라벨 변환
        lbl_path = lbl_dir/(img_path.stem + ".txt")
        new_lbl = DST/f"labels/{split_name_out}/{img_path.stem}.txt"
        if not lbl_path.exists():
            # 이미지에 라벨 없음 → 빈 파일
            new_lbl.write_text("")
            continue

        lines_out = []
        for line in lbl_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls_idx = int(parts[0])
            bbox = parts[1:]
            orig_name = idx2name[cls_idx] if cls_idx < len(idx2name) else str(cls_idx)
            superk = guess_super(orig_name)

            # 만약 라벨명이 키워드와 안 맞으면, 소스 폴더로 대분류 결정
            if superk is None:
                # src_root.path에 따라 강제 매핑
                if "nudity" in str(src_root).lower():
                    superk = "nudity"
                elif "violence" in str(src_root).lower():
                    superk = "violence"
                elif "selfharm" in str(src_root).lower():
                    superk = "selfharm"
                elif "weapon" in str(src_root).lower() or "crime" in str(src_root).lower():
                    superk = "weapon"

            if superk not in FINAL_INDEX:
                # 매핑 실패 → 라벨 드롭(혹은 필요시 continue 대신 특정 클래스에 모을 수도 있음)
                continue

            new_cls = FINAL_INDEX[superk]
            lines_out.append(" ".join([str(new_cls)] + bbox))

        new_lbl.write_text("\n".join(lines_out))

def main():
    for super_name, src in SRC_DIRS.items():
        # Roboflow export 마다 split 폴더명이 train/valid 또는 train/val 다를 수 있음
        if (src/"train").exists() and (src/"valid").exists():
            convert_split(src, "train", "train")
            convert_split(src, "valid", "val")
        elif (src/"train").exists() and (src/"val").exists():
            convert_split(src, "train", "train")
            convert_split(src, "val", "val")
        else:
            print(f"[WARN] No train/val in {src}")

    print(" Merged and remapped to 4 super classes:", FINAL_CLASSES)

if __name__ == "__main__":
    main()
