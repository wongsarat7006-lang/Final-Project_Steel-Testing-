"""
Tier 1 (accuracy) — ทำความสะอาด label ของ merged_dataset หลัง merge + resplit

ทำ 2 อย่าง:
  1. ตัดกล่องเสีย (degenerate) — w หรือ h เล็กกว่า --min-side หรือพื้นที่ < --min-area
     (พบ ~19 กล่องใน crack ที่ area ~0)
  2. รวมกล่องของคลาส "texture เต็มภาพ" (ค่าเริ่มต้น: crazing=0, rolled-in_scale=4)
     ให้เหลือ 1 กล่อง/ภาพ = union ของกล่องเดิม
     เหตุผล: NEU-DET annotate 2 คลาสนี้เป็นกล่องย่อยกระจายมั่ว (เฉลี่ย 2-5 กล่อง,
     สัดส่วนไม่คงที่) ทำให้ detector เรียนไม่ได้ → mAP50 ตันที่ ~0.39
     ทั้งสองคลาสจริง ๆ เป็น texture ทั้ง patch → กล่องเดียวครอบทั้งหมดเหมาะกว่า

Source of truth = labels_raw/ (สร้างครั้งแรกจาก labels/ ปัจจุบัน)
รันซ้ำได้เรื่อย ๆ — จะ regenerate labels/ จาก labels_raw/ เสมอ

วิธีใช้:
    python fix_labels.py                     # ทำจริง ทุก split
    python fix_labels.py --dry-run           # ดูผลก่อน ไม่เขียน
    python fix_labels.py --merge-classes 0 4 --min-side 0.004
    python fix_labels.py --restore           # คืน labels/ กลับเป็นของเดิม แล้วจบ

ลำดับ: merge_datasets.py -> resplit_dataset.py -> **fix_labels.py** ->
        make_grayscale_dataset.py -> make_oversampled_list.py -> train.py
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
         "rolled-in_scale", "scratches", "rust", "crack"]


def parse_label(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            rows.append((int(p[0]), *(float(v) for v in p[1:])))
    return rows


def union_box(boxes):
    """boxes: list ของ (xc, yc, w, h) normalized -> (xc, yc, w, h) ของกล่องรวม (clip 0..1)"""
    x1 = min(b[0] - b[2] / 2 for b in boxes)
    y1 = min(b[1] - b[3] / 2 for b in boxes)
    x2 = max(b[0] + b[2] / 2 for b in boxes)
    y2 = max(b[1] + b[3] / 2 for b in boxes)
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def fix_file(rows, merge_classes, min_side, min_area):
    """คืน (new_rows, n_dropped, n_merged_files_delta) จาก rows ดิบ"""
    kept = [r for r in rows if r[3] >= min_side and r[4] >= min_side and r[3] * r[4] >= min_area]
    n_dropped = len(rows) - len(kept)

    out, merged_here = [], 0
    for cid in range(len(NAMES)):
        cls_boxes = [(r[1], r[2], r[3], r[4]) for r in kept if r[0] == cid]
        if not cls_boxes:
            continue
        if cid in merge_classes and len(cls_boxes) > 1:
            out.append((cid, *union_box(cls_boxes)))
            merged_here = 1
        else:
            out.extend((cid, *b) for b in cls_boxes)
    return out, n_dropped, merged_here


def main():
    ap = argparse.ArgumentParser(description="ทำความสะอาด label ของ merged_dataset")
    ap.add_argument("--dataset", default=str(BASE_DIR / "merged_dataset"))
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--merge-classes", nargs="*", type=int, default=[0, 4],
                    help="class id ที่จะรวมกล่องเป็น union ต่อภาพ (ค่าเริ่มต้น 0=crazing 4=rolled-in_scale)")
    ap.add_argument("--min-side", type=float, default=0.004,
                    help="ตัดกล่องที่ w หรือ h (normalized) เล็กกว่านี้")
    ap.add_argument("--min-area", type=float, default=0.0005,
                    help="ตัดกล่องที่พื้นที่ (normalized) เล็กกว่านี้")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true",
                    help="คืน labels/ จาก labels_raw/ แล้วจบ")
    args = ap.parse_args()

    ds = Path(args.dataset)
    merge_classes = set(args.merge_classes or [])

    for split in args.splits:
        lbl_dir = ds / split / "labels"
        raw_dir = ds / split / "labels_raw"
        if not lbl_dir.is_dir():
            print(f"  ข้าม {split} — ไม่พบ {lbl_dir}")
            continue

        if args.restore:
            if raw_dir.is_dir():
                shutil.rmtree(lbl_dir)
                shutil.copytree(raw_dir, lbl_dir)
                print(f"  {split}: คืน labels/ จาก labels_raw/ แล้ว")
            else:
                print(f"  {split}: ไม่มี labels_raw/ — ข้าม")
            continue

        # ครั้งแรก: สำรอง labels/ -> labels_raw/ (ของดั้งเดิม), ครั้งต่อไปอ่านจาก labels_raw/
        if raw_dir.is_dir():
            src_dir = raw_dir
        elif args.dry_run:
            src_dir = lbl_dir  # dry-run: ยังไม่แตะ disk อ่านจาก labels/ ตรง ๆ
        else:
            shutil.copytree(lbl_dir, raw_dir)
            print(f"  {split}: สำรอง labels/ -> labels_raw/ ({len(list(raw_dir.glob('*.txt')))} ไฟล์)")
            src_dir = raw_dir

        before, after = Counter(), Counter()
        tot_dropped = merged_files = 0
        files = sorted(src_dir.glob("*.txt"))
        for f in files:
            rows = parse_label(f)
            for r in rows:
                before[r[0]] += 1
            new_rows, n_drop, n_merged = fix_file(rows, merge_classes, args.min_side, args.min_area)
            for r in new_rows:
                after[r[0]] += 1
            tot_dropped += n_drop
            merged_files += n_merged
            if not args.dry_run:
                out = "".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                              for (c, xc, yc, w, h) in new_rows)
                (lbl_dir / f.name).write_text(out)

        tag = "[dry-run] " if args.dry_run else ""
        print(f"\n{tag}{split}: {len(files)} ไฟล์ | ตัดกล่องเสีย {tot_dropped} | "
              f"รวมกล่อง texture ใน {merged_files} ไฟล์")
        print(f"  {'class':16s}{'instances(before)':>18s}{'instances(after)':>18s}")
        for i, n in enumerate(NAMES):
            mark = "  <- merge" if i in merge_classes else ""
            print(f"  {n:16s}{before[i]:>18d}{after[i]:>18d}{mark}")

    if not args.restore and not args.dry_run:
        # ลบ cache ของ ultralytics กันใช้ label เก่า
        for split in args.splits:
            for c in (ds / split).glob("labels*.cache"):
                c.unlink()
        print("\nเสร็จ — labels/ อัปเดตแล้ว (ต้นฉบับอยู่ใน labels_raw/)")
        print("ถ้าทำ grayscale: python make_grayscale_dataset.py")
        print("ถ้าไม่ทำ grayscale: python make_oversampled_list.py --dataset merged_dataset")


if __name__ == "__main__":
    main()
