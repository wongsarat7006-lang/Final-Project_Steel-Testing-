"""
วัดผลบน "ชุดภาพเหล็กถ่ายจริง" (real_test/) แบบ image-level
เทียบ 2 สถาปัตยกรรม:

  --mode pipeline  : Stage 1 (หา region เหล็ก) -> Stage 2 (YOLO ตรวจตำหนิบนแต่ละ crop)
  --mode baseline  : Stage 2 (YOLO) บนภาพเต็มตรง ๆ ไม่มี Stage 1
  --mode both      : ทำทั้งสอง แล้วพิมพ์ตารางเทียบ (ค่าเริ่มต้น)

ใช้ตอบคำถามวิทยานิพนธ์: "การมี Stage 1 ช่วยจริงไหม เมื่อเทียบกับ YOLO ภาพเต็ม"

โครงสร้างข้อมูลที่ต้องมี:
    real_test/
      images/            ภาพเหล็กถ่ายจริง (.jpg/.png/...)
      labels.csv         header: filename,classes
                         เช่น   photo_001.jpg,rust;scratches
                                photo_002.jpg,none
วิธีใช้:
    python evaluate_real.py
    python evaluate_real.py --mode pipeline --conf 0.35
    python evaluate_real.py --dir real_test --out real_test_results.json
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

import pipeline as P

BASE_DIR = Path(__file__).resolve().parent
CLASSES = P.DEFECT_CLASSES
NAME_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def load_ground_truth(csv_path: Path):
    """คืน dict: filename -> set(class_id).  'none'/ว่าง = ไม่มีตำหนิ (set ว่าง)"""
    gt = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise SystemExit(f"{csv_path} ต้องมี header: filename,classes")
        for row in reader:
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            raw = (row.get("classes") or "").strip().lower()
            ids = set()
            if raw and raw != "none":
                for tok in raw.replace(",", ";").split(";"):
                    tok = tok.strip()
                    if not tok:
                        continue
                    if tok not in NAME_TO_ID:
                        raise SystemExit(
                            f"{csv_path}: คลาส '{tok}' ของ {fn} ไม่รู้จัก "
                            f"(ต้องเป็นหนึ่งใน {CLASSES})"
                        )
                    ids.add(NAME_TO_ID[tok])
            gt[fn] = ids
    return gt


def predict_pipeline(s1, s2, image, conf, device):
    """คืน (set(class_id), meta)  ผ่าน Stage 1 + Stage 2"""
    mask = P.run_stage1(s1, image, device)
    boxes = P.mask_to_boxes(mask)
    metal_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
    metal_found = len(boxes) > 0
    fallback = metal_ratio < 0.05
    if not boxes or fallback:
        h, w = image.shape[:2]
        boxes = boxes + [(0, 0, w, h)]

    preds = set()
    for (x, y, w, h) in boxes:
        crop = image[y:y + h, x:x + w]
        for d in P.run_stage2(s2, crop, conf, device):
            preds.add(NAME_TO_ID[d["class"]])
    return preds, {
        "metal_found": metal_found,
        "metal_ratio": round(metal_ratio, 4),
        "fallback_full_image": fallback,
        "n_regions": len(boxes),
    }


def predict_baseline(s2, image, conf, device):
    """คืน set(class_id)  จาก YOLO บนภาพเต็ม"""
    preds = set()
    for d in P.run_stage2(s2, image, conf, device):
        preds.add(NAME_TO_ID[d["class"]])
    return preds


def score(gt_map, pred_map):
    """image-level TP/FP/FN ต่อคลาส + micro/macro"""
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for fn_name, gt in gt_map.items():
        pred = pred_map.get(fn_name, set())
        for c in range(len(CLASSES)):
            in_p, in_g = c in pred, c in gt
            if in_p and in_g:
                tp[c] += 1
            elif in_p:
                fp[c] += 1
            elif in_g:
                fn[c] += 1

    rows = []
    mtp = mfp = mfn = 0
    for c in range(len(CLASSES)):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"class": CLASSES[c], "tp": tp[c], "fp": fp[c], "fn": fn[c],
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
        mtp += tp[c]; mfp += fp[c]; mfn += fn[c]

    micro_p = mtp / (mtp + mfp) if mtp + mfp else 0.0
    micro_r = mtp / (mtp + mfn) if mtp + mfn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    # macro เฉลี่ยเฉพาะคลาสที่มีอยู่จริงใน ground truth
    present = [r for r in rows if r["tp"] + r["fn"] > 0]
    macro_f1 = sum(r["f1"] for r in present) / len(present) if present else 0.0
    return {
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": rows,
    }


def print_report(title, summary, extra=None):
    print(f"\n===== {title} =====")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
    s = summary
    print(f"  micro  P={s['micro_precision']}  R={s['micro_recall']}  F1={s['micro_f1']}")
    print(f"  macro-F1 (เฉพาะคลาสที่มีใน GT) = {s['macro_f1']}")
    print(f"  {'class':<18}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>8}{'R':>8}{'F1':>8}")
    for r in s["per_class"]:
        if r["tp"] + r["fp"] + r["fn"] == 0:
            continue
        print(f"  {r['class']:<18}{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}"
              f"{r['precision']:>8}{r['recall']:>8}{r['f1']:>8}")


def main():
    ap = argparse.ArgumentParser(description="วัดผลบนชุดภาพเหล็กถ่ายจริง")
    ap.add_argument("--dir", default=str(BASE_DIR / "real_test"),
                    help="โฟลเดอร์ที่มี images/ และ labels.csv")
    ap.add_argument("--mode", default="both", choices=["pipeline", "baseline", "both"])
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=str(BASE_DIR / "real_test_results.json"))
    args = ap.parse_args()

    root = Path(args.dir)
    img_dir = root / "images"
    csv_path = root / "labels.csv"
    if not img_dir.is_dir() or not csv_path.is_file():
        raise SystemExit(
            f"ยังไม่มีชุดทดสอบภาพจริง\n"
            f"  ต้องสร้าง: {img_dir}\\  (ภาพเหล็กถ่ายจริง 40-60 ภาพ)\n"
            f"            {csv_path}  (header: filename,classes ; 'none' = ไม่มีตำหนิ)\n"
            f"  ดูรายละเอียดใน prepare_data.md หัวข้อ 6"
        )

    gt_map = load_ground_truth(csv_path)
    images = []
    for fn in gt_map:
        p = img_dir / fn
        if not p.exists():
            print(f"  เตือน: ไม่พบไฟล์ภาพ {p} (ข้าม)")
            continue
        images.append(p)
    if not images:
        raise SystemExit("ไม่พบไฟล์ภาพที่ตรงกับ labels.csv เลย")

    n_gt_defect = sum(1 for v in gt_map.values() if v)
    print(f"ชุดทดสอบ: {len(images)} ภาพ  ({n_gt_defect} ภาพมีตำหนิ, "
          f"{len(images) - n_gt_defect} ภาพปกติ)")

    device = P.resolve_device(args.device)
    s1, s2 = P.load_models(device)
    gt_used = {p.name: gt_map[p.name] for p in images}

    report = {"device": device, "conf": args.conf, "n_images": len(images)}

    if args.mode in ("pipeline", "both"):
        pred_map = {}
        meta_all = []
        t0 = time.perf_counter()
        for p in images:
            im = cv2.imread(str(p))
            preds, meta = predict_pipeline(s1, s2, im, args.conf, device)
            pred_map[p.name] = preds
            meta_all.append(meta)
        dt = time.perf_counter() - t0
        summ = score(gt_used, pred_map)
        extra = {
            "stage1_metal_found_rate":
                round(sum(m["metal_found"] for m in meta_all) / len(meta_all), 4),
            "fallback_full_image_rate":
                round(sum(m["fallback_full_image"] for m in meta_all) / len(meta_all), 4),
            "avg_regions_per_image":
                round(sum(m["n_regions"] for m in meta_all) / len(meta_all), 2),
            "sec_per_image": round(dt / len(images), 3),
        }
        report["pipeline"] = {**summ, **extra}
        print_report("PIPELINE (Stage 1 + Stage 2)", summ, extra)

    if args.mode in ("baseline", "both"):
        pred_map = {}
        t0 = time.perf_counter()
        for p in images:
            im = cv2.imread(str(p))
            pred_map[p.name] = predict_baseline(s2, im, args.conf, device)
        dt = time.perf_counter() - t0
        summ = score(gt_used, pred_map)
        extra = {"sec_per_image": round(dt / len(images), 3)}
        report["baseline"] = {**summ, **extra}
        print_report("BASELINE (YOLO บนภาพเต็ม ไม่มี Stage 1)", summ, extra)

    if args.mode == "both":
        p, b = report["pipeline"], report["baseline"]
        print("\n===== เทียบ =====")
        print(f"  {'metric':<14}{'pipeline':>12}{'baseline':>12}{'ต่าง':>10}")
        for k in ("micro_precision", "micro_recall", "micro_f1", "macro_f1"):
            print(f"  {k:<14}{p[k]:>12}{b[k]:>12}{p[k] - b[k]:>+10.4f}")
        print(f"  {'sec/image':<14}{p['sec_per_image']:>12}{b['sec_per_image']:>12}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nบันทึกรายงาน: {args.out}")


if __name__ == "__main__":
    main()
