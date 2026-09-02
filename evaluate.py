"""
วัดผลระบบตรวจตำหนิเหล็ก 2 แบบ:

  --mode stage2    : วัด YOLO Stage 2 ตรงๆ (mAP50 / mAP50-95 ต่อคลาส) ผ่าน ultralytics
  --mode pipeline  : วัดทั้ง pipeline (Stage 1 + Stage 2) แบบ image-level
                     เทียบ "ชนิดตำหนิที่ระบบตอบ" กับ "ชนิดตำหนิใน label" ของแต่ละภาพ
  --mode both      : ทำทั้งสอง (ค่าเริ่มต้น)

วิธีใช้:
    python evaluate.py
    python evaluate.py --mode pipeline --images merged_dataset/test/images --labels merged_dataset/test/labels
    python evaluate.py --mode stage2 --data merged_dataset/data.yaml --split test
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch

import pipeline as P

BASE_DIR = Path(__file__).resolve().parent


# ---------- Stage 2 (ultralytics) ----------
def eval_stage2(data_yaml, split, device):
    from ultralytics import YOLO

    model = YOLO(str(P.STAGE2_MODEL_PATH))
    metrics = model.val(data=str(data_yaml), split=split, device=device, verbose=False)

    names = model.names
    rows = []
    for i, c in enumerate(metrics.box.ap_class_index):
        rows.append({
            "class": names[int(c)],
            "precision": round(float(metrics.box.p[i]), 4),
            "recall": round(float(metrics.box.r[i]), 4),
            "mAP50": round(float(metrics.box.ap50[i]), 4),
            "mAP50-95": round(float(metrics.box.ap[i]), 4),
        })

    result = {
        "split": split,
        "overall": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class": rows,
    }
    print("\n===== Stage 2 (YOLO) =====")
    print(f"  split: {split}")
    print(f"  mAP50 = {result['overall']['mAP50']}   mAP50-95 = {result['overall']['mAP50-95']}")
    print(f"  {'class':<18}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for r in rows:
        print(f"  {r['class']:<18}{r['precision']:>8}{r['recall']:>8}{r['mAP50']:>9}{r['mAP50-95']:>10}")
    return result


# ---------- Pipeline end-to-end (image-level) ----------
def _gt_classes(label_path: Path):
    if not label_path.exists():
        return set()
    classes = set()
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            classes.add(int(parts[0]))
    return classes


def eval_pipeline(images_dir, labels_dir, device, conf):
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    s1, s2 = P.load_models(device)

    n_classes = len(P.DEFECT_CLASSES)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    n_images = 0
    n_metal_found = 0

    images = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in P.IMAGE_EXTS]
    print(f"\n===== Pipeline end-to-end =====")
    print(f"  ภาพทดสอบ: {len(images)} ภาพ")

    for idx, img_path in enumerate(images, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        n_images += 1

        mask = P.run_stage1(s1, image, device)
        # ใช้ fallback แบบเดียวกับ pipeline.py (เพิ่มกรอบทั้งภาพเมื่อ metal_ratio < 0.05)
        boxes, meta = P.build_regions(mask, image.shape)
        if meta["metal_found"]:
            n_metal_found += 1

        pred_classes = set()
        for (x, y, w, h) in boxes:
            crop = image[y:y + h, x:x + w]
            for d in P.run_stage2(s2, crop, conf, device):
                pred_classes.add(P.DEFECT_CLASSES.index(d["class"]))

        gt_classes = _gt_classes(labels_dir / f"{img_path.stem}.txt")

        for c in range(n_classes):
            in_pred, in_gt = c in pred_classes, c in gt_classes
            if in_pred and in_gt:
                tp[c] += 1
            elif in_pred and not in_gt:
                fp[c] += 1
            elif not in_pred and in_gt:
                fn[c] += 1

        if idx % 50 == 0:
            print(f"  ...{idx}/{len(images)}")

    rows = []
    micro_tp = micro_fp = micro_fn = 0
    for c in range(n_classes):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({
            "class": P.DEFECT_CLASSES[c],
            "tp": tp[c], "fp": fp[c], "fn": fn[c],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        })
        micro_tp += tp[c]; micro_fp += fp[c]; micro_fn += fn[c]

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro_f1 = sum(r["f1"] for r in rows) / len(rows)

    result = {
        "images": n_images,
        "stage1_metal_detection_rate": round(n_metal_found / n_images, 4) if n_images else 0.0,
        "image_level": {
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "per_class": rows,
    }

    print(f"  Stage 1 เจอเหล็ก: {n_metal_found}/{n_images} ภาพ ({result['stage1_metal_detection_rate']:.0%})")
    print(f"  image-level  micro-F1 = {result['image_level']['micro_f1']}   "
          f"macro-F1 = {result['image_level']['macro_f1']}")
    print(f"  {'class':<18}{'TP':>5}{'FP':>5}{'FN':>5}{'P':>8}{'R':>8}{'F1':>8}")
    for r in rows:
        print(f"  {r['class']:<18}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
              f"{r['precision']:>8}{r['recall']:>8}{r['f1']:>8}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate steel defect system")
    parser.add_argument("--mode", default="both", choices=["stage2", "pipeline", "both"])
    parser.add_argument("--data", default=str(BASE_DIR / "merged_dataset" / "data.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--images", default=str(BASE_DIR / "merged_dataset" / "test" / "images"))
    parser.add_argument("--labels", default=str(BASE_DIR / "merged_dataset" / "test" / "labels"))
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--weights", default=None,
                        help="path ของ Stage 2 .pt (default: ตัวใน pipeline.py = train-clean/best.pt)")
    parser.add_argument("--out", default=str(BASE_DIR / "evaluation_results.json"))
    args = parser.parse_args()

    if args.weights:
        w = Path(args.weights)
        if not w.is_file():
            raise SystemExit(f"ไม่พบไฟล์ weights: {w}")
        P.STAGE2_MODEL_PATH = w
        print(f"ใช้ Stage 2 weights: {w}")

    device = P.resolve_device(args.device)
    report = {"device": device, "stage2_weights": str(P.STAGE2_MODEL_PATH)}

    if args.mode in ("stage2", "both"):
        report["stage2"] = eval_stage2(args.data, args.split, device)
    if args.mode in ("pipeline", "both"):
        report["pipeline"] = eval_pipeline(args.images, args.labels, device, args.conf)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nบันทึกรายงานที่: {args.out}")


if __name__ == "__main__":
    main()
