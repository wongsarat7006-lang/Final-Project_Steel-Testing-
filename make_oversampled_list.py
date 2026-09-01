"""
สร้างไฟล์ list สำหรับ train แบบ class-balanced oversampling
เพื่อแก้ปัญหาคลาสที่อ่อน (crazing, rolled-in_scale) ที่โมเดล "มองข้าม" ไปเป็น background

หลักการ: ทำซ้ำ path ของรูปในไฟล์ list ตามจำนวนรอบของคลาสที่อยู่ในรูปนั้น
(รูปที่มีหลายคลาส ใช้ตัวคูณสูงสุด) — ไม่ก็อปไฟล์จริง ย้อนกลับได้ทันที

ผลลัพธ์: merged_dataset/train_oversampled.txt
ใช้คู่กับ data_oversampled.yaml (train: train_oversampled.txt)

วิธีใช้:
    python make_oversampled_list.py
    python make_oversampled_list.py --show     # ดูจำนวน instance ก่อน/หลัง เฉย ๆ
"""
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
DS_DIR = BASE_DIR / "merged_dataset"
IMG_DIR = DS_DIR / "train" / "images"
LBL_DIR = DS_DIR / "train" / "labels"
OUT_TXT = DS_DIR / "train_oversampled.txt"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
         "rolled-in_scale", "scratches", "rust", "crack"]

# ตัวคูณ oversampling ต่อคลาส — ตั้งจาก confusion matrix (val-2):
#   crazing 0.64 หลุดเป็น background, rolled-in_scale 0.34, scratches/pitted แนวโน้มพลาดบ้าง
CLASS_MULT = {
    0: 3,   # crazing        555 -> ~1665
    1: 1,   # inclusion      783
    2: 1,   # patches        701
    3: 2,   # pitted_surface 349 -> ~698
    4: 2,   # rolled-in_scale 504 -> ~1008
    5: 2,   # scratches      444 -> ~888
    6: 1,   # rust           705
    7: 1,   # crack          1125
}


def classes_in(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    out = set()
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            out.add(int(parts[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="แสดงสถิติอย่างเดียว ไม่เขียนไฟล์")
    args = ap.parse_args()

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"ไม่พบรูปใน {IMG_DIR}")

    lines: list[str] = []
    before, after = Counter(), Counter()
    dup_imgs = 0

    for img in imgs:
        lbl = LBL_DIR / (img.stem + ".txt")
        cls = classes_in(lbl)
        rep = max((CLASS_MULT.get(c, 1) for c in cls), default=1)
        for c in cls:
            before[c] += 1
            after[c] += rep
        if rep > 1:
            dup_imgs += 1
        # path แบบ absolute เพื่อกันปัญหา relative resolution ของ ultralytics
        lines.extend([str(img.resolve())] * rep)

    print(f"รูป train ทั้งหมด : {len(imgs)}")
    print(f"รูปที่ถูก oversample : {dup_imgs}")
    print(f"จำนวนบรรทัดในไฟล์ list : {len(lines)}  (จากเดิม {len(imgs)})\n")
    print(f"{'class':16s} {'images(before)':>15s} {'images(after)':>15s}  x")
    for i, n in enumerate(NAMES):
        print(f"{n:16s} {before[i]:>15d} {after[i]:>15d}  {CLASS_MULT[i]}")

    if args.show:
        return

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"\nเขียนแล้ว: {OUT_TXT}")
    print("ต่อไป: python train.py --recipe texture --data data_oversampled.yaml --name train-balanced --epochs 120")


if __name__ == "__main__":
    main()
