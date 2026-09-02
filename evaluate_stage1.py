"""
วัดผล Stage 1 (DMS46 Metal Localization) เชิงตัวเลข — ทำงานบนชุดข้อมูล YOLO ที่มีอยู่แล้ว
โดยถือว่า "ทุกภาพในชุดนี้เป็นเหล็ก" (merged_dataset เป็นภาพ close-up ผิวเหล็กทั้งหมด)

ตอบคำถาม ablation: "Stage 1 คุ้มไหม / มันตัดตำหนิจริงทิ้งหรือเปล่า"

เมตริกต่อภาพ (คิดจากกรอบ metal ก่อน fallback):
  metal_found        เจอ region เหล็กอย่างน้อย 1 กรอบ
  fallback           metal_ratio < min-metal-ratio  (Stage 1 แทบไม่ช่วย ต้องตรวจทั้งภาพ)
  metal_ratio        สัดส่วนพิกเซลที่ DMS ทำนายว่าเป็น Metal
  n_regions          จำนวนกรอบหลังรวมกรอบติดกัน
  box_coverage       สัดส่วนพื้นที่ภาพที่กรอบ metal (union) ครอบคลุม
                     ~1.0 = Stage 1 คืนเกือบทั้งภาพ (ไม่ได้ crop อะไร)
  gt_center_inside   สัดส่วนกล่อง GT ตำหนิที่ "จุดกึ่งกลาง" ตกในกรอบ metal
  gt_area_kept       เฉลี่ย (พื้นที่กล่อง GT ที่อยู่ในกรอบ metal / พื้นที่กล่อง GT)
                     < 1.0 = Stage 1 crop ทับตำหนิจริงบางส่วนทิ้ง -> เสีย recall
  stage1_ms          เวลาเฉพาะ Stage 1 ต่อภาพ (ต้นทุนที่จ่ายเพิ่ม)

วิธีใช้:
    python evaluate_stage1.py
    python evaluate_stage1.py --dir merged_dataset/valid --device cpu
    python evaluate_stage1.py --dir merged_dataset/test --limit 100 --out results/stage1_dms46_test.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import pipeline as P

BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """อ่าน YOLO label (cls cx cy w h normalized) -> list ของ (x1, y1, x2, y2) พิกเซล"""
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = (float(v) for v in parts[:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def metal_box_mask(boxes, img_w: int, img_h: int):
    """รวมกรอบ metal (x, y, w, h) เป็น binary mask ขนาดภาพ สำหรับคิดพื้นที่ union"""
    m = np.zeros((img_h, img_w), np.uint8)
    for (x, y, w, h) in boxes:
        m[max(0, y):y + h, max(0, x):x + w] = 1
    return m


def eval_image(s1, image, gt_boxes, device, min_metal_ratio):
    h, w = image.shape[:2]
    t0 = time.perf_counter()
    mask = P.run_stage1(s1, image, device)
    stage1_ms = (time.perf_counter() - t0) * 1000

    boxes = P.mask_to_boxes(mask)
    metal_ratio = cv2.countNonZero(mask) / (h * w)
    box_m = metal_box_mask(boxes, w, h)
    box_coverage = float(box_m.sum()) / (h * w)

    center_hits = area_kept = 0.0
    for (gx1, gy1, gx2, gy2) in gt_boxes:
        gcx = int(np.clip((gx1 + gx2) / 2, 0, w - 1))
        gcy = int(np.clip((gy1 + gy2) / 2, 0, h - 1))
        center_hits += int(box_m[gcy, gcx] > 0)

        ix1, iy1 = int(max(0, gx1)), int(max(0, gy1))
        ix2, iy2 = int(min(w, gx2)), int(min(h, gy2))
        gt_area = max(1.0, (gx2 - gx1) * (gy2 - gy1))
        inside = float(box_m[iy1:iy2, ix1:ix2].sum()) if ix2 > ix1 and iy2 > iy1 else 0.0
        area_kept += inside / gt_area

    n_gt = len(gt_boxes)
    return {
        "metal_found": len(boxes) > 0,
        "fallback": metal_ratio < min_metal_ratio,
        "metal_ratio": metal_ratio,
        "n_regions": len(boxes),
        "box_coverage": box_coverage,
        "n_gt": n_gt,
        "gt_center_inside": (center_hits / n_gt) if n_gt else None,
        "gt_area_kept": (area_kept / n_gt) if n_gt else None,
        "stage1_ms": stage1_ms,
    }


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def median(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return 0.0
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def main():
    ap = argparse.ArgumentParser(description="วัดผล Stage 1 (DMS46) เชิงตัวเลข")
    ap.add_argument("--dir", default=str(BASE_DIR / "merged_dataset" / "test"),
                    help="โฟลเดอร์ YOLO ที่มี images/ และ labels/")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--min-metal-ratio", type=float, default=0.05,
                    help="เกณฑ์ fallback เดียวกับ pipeline.py")
    ap.add_argument("--limit", type=int, default=0, help="จำกัดจำนวนภาพ (0 = ทั้งหมด)")
    ap.add_argument("--out", default=str(BASE_DIR / "results" / "stage1_latest.json"))
    args = ap.parse_args()

    root = Path(args.dir)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"ไม่พบ {img_dir}")

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"ไม่พบไฟล์ภาพใน {img_dir}")

    device = P.resolve_device(args.device)
    print(f"โหลด Stage 1 (device={device})...")
    s1, _ = P.load_models(device)

    print(f"ประเมิน {len(images)} ภาพจาก {root}\n")
    per_image = []
    for i, p in enumerate(images, 1):
        image = cv2.imread(str(p))
        if image is None:
            continue
        gt = load_gt_boxes(lbl_dir / f"{p.stem}.txt", image.shape[1], image.shape[0])
        r = eval_image(s1, image, gt, device, args.min_metal_ratio)
        r["file"] = p.name
        per_image.append(r)
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)}")

    n = len(per_image)
    with_gt = [r for r in per_image if r["n_gt"] > 0]
    agg = {
        "dir": str(root),
        "device": device,
        "n_images": n,
        "n_images_with_gt": len(with_gt),
        "metal_found_rate": round(mean(r["metal_found"] for r in per_image), 4),
        "fallback_rate": round(mean(r["fallback"] for r in per_image), 4),
        "metal_ratio_mean": round(mean(r["metal_ratio"] for r in per_image), 4),
        "metal_ratio_median": round(median([r["metal_ratio"] for r in per_image]), 4),
        "box_coverage_mean": round(mean(r["box_coverage"] for r in per_image), 4),
        "avg_regions_per_image": round(mean(r["n_regions"] for r in per_image), 2),
        "gt_center_inside_rate": round(mean(r["gt_center_inside"] for r in with_gt), 4),
        "gt_area_kept_mean": round(mean(r["gt_area_kept"] for r in with_gt), 4),
        "stage1_ms_mean": round(mean(r["stage1_ms"] for r in per_image), 1),
        "stage1_ms_median": round(median([r["stage1_ms"] for r in per_image]), 1),
    }

    print("\n===== สรุป Stage 1 =====")
    for k, v in agg.items():
        print(f"  {k:<24} {v}")

    print("\n----- อ่านผล -----")
    if agg["box_coverage_mean"] > 0.9:
        print("  • box_coverage ~1.0 : Stage 1 คืนเกือบทั้งภาพบน close-up เหล็ก")
        print("    -> บนภาพชนิดนี้ Stage 1 แทบไม่ได้ crop = ประโยชน์หลักอยู่ที่ 'กันภาพที่ไม่ใช่เหล็ก'")
    if agg["fallback_rate"] > 0.2:
        print(f"  • fallback_rate {agg['fallback_rate']:.0%} : หลายภาพต้องตรวจทั้งภาพอยู่ดี")
    if agg["gt_area_kept_mean"] < 0.98:
        print(f"  • gt_area_kept {agg['gt_area_kept_mean']:.1%} < 100% : Stage 1 crop ทับตำหนิจริงบางส่วนทิ้ง")
        print("    -> เป็นความเสี่ยงต่อ recall ควรรายงานใน Limitations")
    else:
        print(f"  • gt_area_kept {agg['gt_area_kept_mean']:.1%} : Stage 1 เก็บพื้นที่ตำหนิไว้เกือบครบ ไม่กระทบ recall")
    print(f"  • ต้นทุนเวลา Stage 1 = {agg['stage1_ms_mean']} ms/ภาพ (median {agg['stage1_ms_median']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"aggregate": agg, "per_image": per_image}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nบันทึกรายงาน: {args.out}")


if __name__ == "__main__":
    main()
