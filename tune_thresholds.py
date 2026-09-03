"""
Tier 2 (accuracy) — หา confidence threshold ที่ดีที่สุด "รายคลาส" จาก val split

ปัญหา: pipeline ใช้ conf ค่าเดียว (0.4) ทุกคลาส แต่แต่ละคลาสจุดทำงานที่ดีต่างกันมาก
(เช่น rust มั่นใจสูงมากตลอด → ตั้งสูงได้, crazing ยิงเบา ๆ → ต้องตั้งต่ำ)

วิธี: รัน model.val() เอา F1-vs-confidence curve รายคลาส แล้วเลือก conf ที่ F1 สูงสุด
เขียน thresholds.json → pipeline.py / app.py / evaluate_real.py จะโหลดไปใช้อัตโนมัติถ้ามีไฟล์

วิธีใช้:
    python tune_thresholds.py --weights runs/detect/train-gray-s/weights/best.pt \
                              --data merged_dataset_gray/data.yaml
    python tune_thresholds.py --floor 0.2 --ceil 0.9      # จำกัดช่วง conf ที่ยอมรับ

รันหลังเทรนโมเดลใหม่เสร็จ (Tier 1 + yolo11s)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="หา per-class confidence threshold จาก val")
    ap.add_argument("--weights", default="runs/detect/train-gray-s/weights/best.pt")
    ap.add_argument("--data", default="merged_dataset_gray/data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--floor", type=float, default=0.15, help="conf ต่ำสุดที่ยอมรับ")
    ap.add_argument("--ceil", type=float, default=0.92, help="conf สูงสุดที่ยอมรับ")
    ap.add_argument("--compare-conf", type=float, default=0.4, help="conf เดิมที่ใช้เทียบ")
    ap.add_argument("--out", default=str(BASE_DIR / "thresholds.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    w = Path(args.weights)
    if not w.is_file():
        raise SystemExit(f"ไม่พบ weights: {w}")
    data = args.data if os.path.isabs(args.data) else os.path.abspath(args.data)

    device = args.device
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(w))
    m = model.val(data=data, split=args.split, device=device, verbose=False, plots=False)
    b = m.box
    px = np.asarray(b.px)                    # (1000,) ค่า confidence 0..1
    f1c = np.asarray(b.f1_curve)             # (nc, 1000)
    pc = np.asarray(b.p_curve)
    rc = np.asarray(b.r_curve)

    def at(curve_row, conf):
        return float(curve_row[int(np.clip(np.searchsorted(px, conf), 0, len(px) - 1))])

    per_class = {}
    print(f"\n{'class':16s}{'best_conf':>10s}{'F1@best':>9s}{'F1@%.2f'%args.compare_conf:>9s}"
          f"{'P@best':>8s}{'R@best':>8s}")
    for i, ci in enumerate(b.ap_class_index):
        name = m.names[int(ci)]
        row = f1c[i]
        j = int(np.argmax(row))
        best_conf = float(np.clip(px[j], args.floor, args.ceil))
        j_clamped = int(np.clip(np.searchsorted(px, best_conf), 0, len(px) - 1))
        rec = {
            "conf": round(best_conf, 3),
            "f1_opt": round(float(row[j_clamped]), 4),
            "f1_at_compare": round(at(row, args.compare_conf), 4),
            "precision_opt": round(float(pc[i][j_clamped]), 4),
            "recall_opt": round(float(rc[i][j_clamped]), 4),
        }
        per_class[name] = rec
        print(f"{name:16s}{rec['conf']:>10.3f}{rec['f1_opt']:>9.3f}"
              f"{rec['f1_at_compare']:>9.3f}{rec['precision_opt']:>8.3f}{rec['recall_opt']:>8.3f}")

    macro_opt = np.mean([v["f1_opt"] for v in per_class.values()])
    macro_cmp = np.mean([v["f1_at_compare"] for v in per_class.values()])
    out = {
        "weights": str(w),
        "data": data,
        "split": args.split,
        "compare_conf": args.compare_conf,
        "macro_f1_opt": round(float(macro_opt), 4),
        "macro_f1_at_compare": round(float(macro_cmp), 4),
        "per_class": per_class,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmacro-F1: {macro_cmp:.3f} (conf {args.compare_conf})  ->  {macro_opt:.3f} (per-class)")
    print(f"เขียน: {args.out}")
    print("pipeline.py / app.py / evaluate_real.py จะใช้ไฟล์นี้อัตโนมัติ "
          "(ปิดด้วย --no-class-conf ใน pipeline.py)")


if __name__ == "__main__":
    main()
