"""
แบ่ง train/valid/test ของ merged_dataset ใหม่แบบ stratified ให้ทุก split มีครบทุกคลาส

ปัญหาเดิม: test split ของ merged_dataset มี instance แค่คลาส 0,1,2,6,7
           (ไม่มี pitted_surface, rolled-in_scale, scratches) ทำให้วัดผลไม่ครบ

วิธีใช้:
    python resplit_dataset.py                       # 80/10/10, seed 0
    python resplit_dataset.py --val 0.15 --test 0.15
    python resplit_dataset.py --dry-run             # ดูผลก่อน ไม่ย้ายไฟล์จริง

หลังรันเสร็จให้เทรนใหม่:  python train.py
"""
import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "merged_dataset"
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gather():
    """คืน list ของ (image_path, label_path, primary_class) จากทุก split ปัจจุบัน"""
    items = []
    for split in SPLITS:
        img_dir = DATASET / split / "images"
        lbl_dir = DATASET / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            classes = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if p:
                        classes.append(int(p[0]))
            # จัดกลุ่มตามคลาสที่ "พบน้อยสุดในภาพนี้" เพื่อให้คลาสหายากกระจายทั่วถึง
            primary = min(classes, key=lambda c: GLOBAL_FREQ[c]) if classes else -1
            items.append((img, lbl, primary))
    return items


def main():
    parser = argparse.ArgumentParser(description="Stratified re-split ของ merged_dataset")
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not DATASET.exists():
        raise SystemExit(f"ไม่พบ {DATASET} — รัน merge_datasets.py ก่อน")

    # นับความถี่คลาสรวมทั้ง dataset (ใช้เลือก primary class)
    global GLOBAL_FREQ
    GLOBAL_FREQ = Counter()
    for split in SPLITS:
        lbl_dir = DATASET / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            for line in lbl.read_text().splitlines():
                p = line.split()
                if p:
                    GLOBAL_FREQ[int(p[0])] += 1
    GLOBAL_FREQ = defaultdict(lambda: 1, GLOBAL_FREQ)

    items = gather()
    by_class = defaultdict(list)
    for it in items:
        by_class[it[2]].append(it)

    rng = random.Random(args.seed)
    assign = {}  # image stem -> split
    for cls, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, int(n * args.test)) if n > 2 else 0
        n_val = max(1, int(n * args.val)) if n > 2 else 0
        for i, (img, _lbl, _c) in enumerate(group):
            if i < n_test:
                assign[img] = "test"
            elif i < n_test + n_val:
                assign[img] = "valid"
            else:
                assign[img] = "train"

    # สรุปผล
    dist = {s: Counter() for s in SPLITS}
    counts = Counter()
    for img, lbl, _c in items:
        s = assign[img]
        counts[s] += 1
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if p:
                    dist[s][int(p[0])] += 1

    print("จำนวนภาพต่อ split:", dict(counts))
    for s in SPLITS:
        print(f"  {s:6} class dist: {dict(sorted(dist[s].items()))}")

    if args.dry_run:
        print("\n[dry-run] ไม่ได้ย้ายไฟล์")
        return

    # ย้ายไฟล์: เขียนลงโฟลเดอร์ชั่วคราวก่อน แล้วสลับ
    staging = DATASET / "_restaged"
    if staging.exists():
        shutil.rmtree(staging)
    for s in SPLITS:
        (staging / s / "images").mkdir(parents=True, exist_ok=True)
        (staging / s / "labels").mkdir(parents=True, exist_ok=True)

    for img, lbl, _c in items:
        s = assign[img]
        shutil.copy(img, staging / s / "images" / img.name)
        if lbl.exists():
            shutil.copy(lbl, staging / s / "labels" / lbl.name)
        else:
            (staging / s / "labels" / f"{img.stem}.txt").touch()

    for s in SPLITS:
        for sub in ("images", "labels"):
            old = DATASET / s / sub
            if old.exists():
                shutil.rmtree(old)
            shutil.move(str(staging / s / sub), str(old))
        cache = DATASET / s / "labels.cache"
        if cache.exists():
            cache.unlink()
    shutil.rmtree(staging)

    print("\nแบ่ง split ใหม่เรียบร้อย — เทรนต่อด้วย: python train.py")


if __name__ == "__main__":
    main()
