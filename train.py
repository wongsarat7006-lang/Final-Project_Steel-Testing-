"""
เทรนโมเดล Stage 2 (YOLO11) ตรวจตำหนิพื้นผิวเหล็ก 8 คลาส บน merged_dataset

วิธีใช้:
    python train.py                       # ค่าเริ่มต้น: yolo11n, 50 epochs, imgsz 640
    python train.py --epochs 100 --model yolo11s.pt
    python train.py --data data.yaml      # เทรนเฉพาะ 6 คลาส NEU เดิม
    python train.py --resume

ผลลัพธ์อยู่ใน runs/detect/<name>/  (weights/best.pt ใช้ต่อใน pipeline.py)
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "merged_dataset" / "data.yaml"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 steel-defect detector")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="path ของ data.yaml")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="โมเดลเริ่มต้น (yolo11n/s/m .pt) หรือ path ของ .pt ที่จะเทรนต่อ")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="RTX 3050 6GB: yolo11n ใช้ 16 ได้, yolo11s ลองลดเป็น 8")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "0"])
    parser.add_argument("--name", default="train-merged")
    parser.add_argument("--patience", type=int, default=20, help="early stopping")
    parser.add_argument("--resume", action="store_true", help="เทรนต่อจาก checkpoint ล่าสุด")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--recipe", default="default", choices=["default", "texture"],
                        help="default = augmentation เดิม | "
                             "texture = ลด mosaic/scale เพื่อรักษา texture เต็มภาพ "
                             "(แก้คลาส crazing/rolled-in_scale ที่โดนมองข้าม)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        device = "0"
    else:
        device = args.device

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"ไม่พบ {data_path}\n"
            f"สร้าง merged_dataset ก่อนด้วย: python merge_datasets.py"
        )

    if args.resume:
        # resume: โหลด last.pt ของ run เดิม แล้วให้ ultralytics อ่าน args จาก checkpoint เอง
        ckpt = BASE_DIR / "runs" / "detect" / args.name / "weights" / "last.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"ไม่พบ checkpoint สำหรับ resume: {ckpt}")
        print(f"resume จาก: {ckpt}")
        model = YOLO(str(ckpt))
        results = model.train(resume=True)
    else:
        print(f"data   : {data_path}")
        print(f"model  : {args.model}")
        print(f"device : {device}  (cuda available = {torch.cuda.is_available()})")
        print(f"epochs : {args.epochs}  imgsz {args.imgsz}  batch {args.batch}\n")

        # --- augmentation recipes ---
        # default : mosaic เต็ม + scale/degrees สูง — ดีกับคลาสที่ตำหนิเป็นก้อนชัด
        # texture : ลด mosaic (0.3) + close เร็ว + scale/degrees ต่ำ — รักษา texture ละเอียด
        #           ที่กินทั้งภาพ (crazing = ร่างแหรอยแตก, rolled-in_scale = ริ้วสเกล)
        #           เพราะ mosaic ย่อภาพ 200px เหลือ ~100px ทำให้ texture หายจนโมเดล
        #           เรียนรู้ว่า "ไม่มั่นใจ = background" -> recall ตก (ดู confusion matrix val-2)
        if args.recipe == "texture":
            aug = dict(
                hsv_h=0.015, hsv_s=0.3, hsv_v=0.4,
                degrees=5.0, translate=0.1, scale=0.2, fliplr=0.5, flipud=0.5,
                mosaic=0.3, close_mosaic=20, mixup=0.0, erasing=0.2,
                cos_lr=True,
            )
        else:
            aug = dict(
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
                degrees=10.0, translate=0.1, scale=0.4, fliplr=0.5, flipud=0.2,
                mosaic=1.0, close_mosaic=10,
            )
        print(f"recipe : {args.recipe}  ->  {aug}\n")

        model = YOLO(args.model)
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            name=args.name,
            patience=args.patience,
            workers=args.workers,
            seed=0,
            deterministic=True,
            **aug,
        )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("runs/detect") / args.name
    print(f"\nเทรนเสร็จ — weights: {save_dir / 'weights' / 'best.pt'}")
    print("ต่อไป: อัปเดต STAGE2_MODEL_PATH ใน pipeline.py ให้ชี้มาที่ best.pt นี้ (ถ้าเปลี่ยนชื่อ run)")

    print("\nรัน validation บน test split...")
    metrics = model.val(data=str(data_path), split="test", device=device)
    print(f"  mAP50    = {metrics.box.map50:.4f}")
    print(f"  mAP50-95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
