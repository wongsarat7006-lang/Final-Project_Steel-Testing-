"""
Active-learning bootstrap — ใช้โมเดลปัจจุบัน "ร่าง" label ให้ภาพใหม่ แล้วคนแค่แก้กรอบที่ผิด
(เร็วกว่าวาดจากศูนย์ 3-5 เท่า) — ขั้นแรกของการสร้าง dataset ภาพถ่ายจริง

    python annotate_bootstrap.py --src ภาพใหม่/ --out dataset_real/round1 --conf 0.12

ผลลัพธ์:
    <out>/images/*.jpg      ภาพต้นฉบับ (คัดลอกมา)
    <out>/labels/*.txt      กรอบที่โมเดลเดา (YOLO format) — เปิดในโปรแกรม label แล้วแก้
    <out>/classes.txt       รายชื่อคลาส (สำหรับ LabelImg / Label Studio)
    <out>/data.yaml         config สำหรับเทรนต่อ (หลัง label เสร็จ merge เข้า merged_dataset)

ขั้นตอนถัดไป:
    1. เปิด <out>/ ในโปรแกรม label (LabelImg, Label Studio, หรือ import Roboflow)
       - ลบกรอบที่ผิด, ขยับ/ปรับกรอบที่เพี้ยน, เพิ่มกรอบที่โมเดลพลาด, แก้ชนิด
    2. รวมเข้าชุดเทรน: คัดลอก images/ + labels/ ไปต่อท้าย merged_dataset/train/
       แล้ว python make_oversampled_list.py --dataset merged_dataset
    3. python train.py --recipe camera --data merged_dataset/data_oversampled.yaml --name train-real1
    4. รอบต่อไป: รัน bootstrap อีกด้วยโมเดลใหม่ บนภาพชุดถัดไป — วนแบบนี้
"""
import argparse
import shutil
from pathlib import Path

import cv2

import pipeline as P

BASE = Path(__file__).resolve().parent
CLASSES = P.DEFECT_CLASSES
NAME_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def main():
    ap = argparse.ArgumentParser(description="ร่าง label ภาพใหม่ด้วยโมเดลปัจจุบัน (active learning)")
    ap.add_argument("--src", required=True, help="โฟลเดอร์ภาพถ่ายจริงชุดใหม่")
    ap.add_argument("--out", required=True, help="โฟลเดอร์ปลายทาง (images/ + labels/)")
    ap.add_argument("--weights", default=None, help="Stage 2 .pt (ไม่ระบุ = ใช้ค่าใน pipeline.py)")
    ap.add_argument("--conf", type=float, default=0.12,
                    help="conf ต่ำ ๆ เพื่อดึงกรอบมาให้เยอะไว้ก่อน (คนค่อยลบที่เกิน)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-stage1", action="store_true",
                    help="ข้าม Stage 1 (แนะนำสำหรับภาพลูกค้าที่เล็งชิ้นเดียวเต็มเฟรม)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์ {src}")
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    if args.weights:
        P.STAGE2_MODEL_PATH = Path(args.weights)
    device = P.resolve_device(args.device)
    s1, s2 = P.load_models(device)
    class_conf = None  # ไม่กรองด้วย per-class threshold — อยากได้กรอบเยอะไว้ก่อน

    imgs = [p for p in sorted(src.iterdir()) if p.suffix.lower() in P.IMAGE_EXTS]
    if not imgs:
        raise SystemExit(f"ไม่พบไฟล์ภาพใน {src}")
    print(f"พบ {len(imgs)} ภาพ — ร่าง label ที่ conf ≥ {args.conf}\n")

    total_boxes = 0
    per_class = {c: 0 for c in CLASSES}
    for ip in imgs:
        image = cv2.imread(str(ip))
        if image is None:
            print(f"  ข้าม {ip.name} (เปิดไม่ได้)")
            continue
        H, W = image.shape[:2]

        if args.no_stage1:
            regions = [(0, 0, W, H)]
        else:
            mask = P.run_stage1(s1, image, device)
            regions, _ = P.build_regions(mask, image.shape)

        lines = []
        for (x, y, w, h) in regions:
            crop = image[y:y + h, x:x + w]
            for d in P.run_stage2(s2, crop, args.conf, device, class_conf=class_conf):
                cx1, cy1, cx2, cy2 = d["bbox_xyxy_crop"]
                gx1, gy1, gx2, gy2 = cx1 + x, cy1 + y, cx2 + x, cy2 + y
                xc = (gx1 + gx2) / 2 / W
                yc = (gy1 + gy2) / 2 / H
                bw = (gx2 - gx1) / W
                bh = (gy2 - gy1) / H
                cid = NAME_TO_ID[d["class"]]
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                per_class[d["class"]] += 1

        shutil.copy(ip, out / "images" / ip.name)
        (out / "labels" / f"{ip.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        total_boxes += len(lines)
        print(f"  {ip.name:40s} -> {len(lines)} กรอบ")

    (out / "classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")

    print(f"\nรวม {total_boxes} กรอบ (ร่าง) ใน {len(imgs)} ภาพ")
    print("ต่อคลาส:", {k: v for k, v in per_class.items() if v})
    print(f"\nเปิด {out}/ ในโปรแกรม label (LabelImg / Label Studio) แล้วแก้กรอบ")
    print("จากนั้น merge เข้า merged_dataset/train/ แล้ว retrain — ดูหัวคอมเมนต์ไฟล์นี้")


if __name__ == "__main__":
    main()
