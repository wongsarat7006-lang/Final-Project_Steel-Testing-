"""
train_gate.py — เทรน classifier "พื้นผิวเหล็ก / ไม่ใช่" (Stage 0 ของ pipeline เดโม)

    python make_gate_dataset.py     # สร้าง dataset_gate/ ก่อน
    python train_gate.py            # ~15-25 นาที บน GPU

ผลลัพธ์: runs/classify/steel-gate/weights/best.pt  -> steel_gate.py / app.py หยิบไปใช้เอง
เล่มจบไม่ใช้ตัวนี้ (pipeline.py ไม่เกี่ยว) — เป็นส่วนของเดโม/โปรดักต์
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--name", default="steel-gate")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    data = BASE / "dataset_gate"
    if not (data / "train").is_dir():
        raise SystemExit("ไม่พบ dataset_gate/ — รัน python make_gate_dataset.py ก่อน")

    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, name=args.name, device=args.device,
                patience=6, pretrained=True,
                # gate ต้องทน "ภาพถ่ายจริง" -> aug แสง/มุม/เบลอ หนักหน่อย
                hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=15, translate=0.12,
                scale=0.5, fliplr=0.5, erasing=0.3)

    best = BASE / "runs" / "classify" / args.name / "weights" / "best.pt"
    print(f"\nเสร็จ -> {best}")
    if best.exists():
        m = YOLO(str(best))
        r = m.val(data=str(data), split="val", device=args.device)
        print("val top-1:", getattr(getattr(r, "top1", None), "__float__", lambda: r)() if r else "?")
        print("แก้ steel_gate.py ให้ชี้ weights นี้ได้เลย (default ชี้อยู่แล้ว) แล้วรัน app.py")


if __name__ == "__main__":
    main()
