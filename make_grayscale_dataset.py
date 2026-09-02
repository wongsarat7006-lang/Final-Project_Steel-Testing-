"""
Tier 1 (accuracy) — สร้าง merged_dataset_gray/ = merged_dataset/ แบบ grayscale (3 ช่อง)

เหตุผล: rust/scratches แม่นเกือบ 100% ส่วนหนึ่งเพราะโมเดลแยก "NEU (เทา) vs rust/crack (สี)"
จาก "โทนสีของภาพ" ไม่ใช่ลักษณะตำหนิ → บนภาพถ่ายจริงจะพัง
เทรนบน grayscale บังคับให้เรียนจาก "รูปร่าง/พื้นผิวของตำหนิ" อย่างเดียว → generalize ดีขึ้น

- ภาพ: แปลงเป็น grayscale แล้ว replicate เป็น 3 ช่อง (ให้เข้า YOLO ปกติ), เขียนเป็น .jpg คุณภาพ 95
- label: copy ตรง ๆ (ไม่เปลี่ยน)
- สร้าง merged_dataset_gray/data.yaml (path ชี้มาที่โฟลเดอร์ gray)

วิธีใช้:
    python make_grayscale_dataset.py
    python make_grayscale_dataset.py --src merged_dataset --dst merged_dataset_gray

ลำดับ: merge_datasets.py -> resplit_dataset.py -> fix_labels.py ->
        **make_grayscale_dataset.py** -> make_oversampled_list.py --dataset merged_dataset_gray -> train.py
"""
import argparse
import shutil
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
         "rolled-in_scale", "scratches", "rust", "crack"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser(description="สร้าง dataset เวอร์ชัน grayscale")
    ap.add_argument("--src", default=str(BASE_DIR / "merged_dataset"))
    ap.add_argument("--dst", default=str(BASE_DIR / "merged_dataset_gray"))
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"ไม่พบ {src} — รัน merge_datasets.py ก่อน")

    for split in args.splits:
        s_img, s_lbl = src / split / "images", src / split / "labels"
        d_img, d_lbl = dst / split / "images", dst / split / "labels"
        if not s_img.is_dir():
            print(f"  ข้าม {split} — ไม่พบ {s_img}")
            continue
        if d_img.exists():
            shutil.rmtree(d_img)
        if d_lbl.exists():
            shutil.rmtree(d_lbl)
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)

        imgs = [p for p in sorted(s_img.iterdir()) if p.suffix.lower() in IMG_EXTS]
        n_ok = 0
        for p in imgs:
            im = cv2.imread(str(p))
            if im is None:
                print(f"    อ่านไม่ได้ (ข้าม): {p.name}")
                continue
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            out_name = p.stem + ".jpg"
            cv2.imwrite(str(d_img / out_name), gray3,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            lbl = s_lbl / (p.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, d_lbl / lbl.name)
            else:
                (d_lbl / (p.stem + ".txt")).write_text("")
            n_ok += 1
        print(f"  {split}: {n_ok}/{len(imgs)} ภาพ -> {d_img}")

    abs_root = str(dst.resolve()).replace("\\", "/")
    (dst / "data.yaml").write_text(
        f"path: {abs_root}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    print(f"\nเสร็จ — {dst}/data.yaml")
    print(f"ต่อไป: python make_oversampled_list.py --dataset {dst.name}")


if __name__ == "__main__":
    main()
