"""
ตรวจ data leakage — ภาพซ้ำ/เกือบซ้ำที่คร่อม split (train↔valid↔test)

ใช้ perceptual hash (dHash 8x8 + aHash) เทียบทุกคู่ข้าม split ถ้า Hamming distance
ต่ำกว่าเกณฑ์ = สงสัยว่าเป็นภาพเดียวกัน/ซ้อนกัน -> รั่ว -> ตัวเลข val/test เชื่อไม่ได้

    python check_leakage.py --data merged_dataset_gray
    python check_leakage.py --data merged_dataset_gray --threshold 6 --out results/leakage_gray.json

ไม่มี dependency นอกจาก Pillow + numpy (ที่ requirements มีอยู่แล้ว)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ชื่อคลาสจากชื่อไฟล์ (roboflow ใส่ prefix ไว้) — ใช้บังคับว่าเทียบเฉพาะภาพคลาสเดียวกัน
_CLASSES = ["neu_crazing", "neu_inclusion", "neu_patches", "neu_pitted_surface",
            "neu_rolled-in_scale", "neu_scratches", "crazing", "inclusion", "patches",
            "pitted_surface", "rolled-in_scale", "scratches", "rust", "crack"]


def class_of(name):
    for c in _CLASSES:
        if name.startswith(c + "_") or name.startswith(c + "-"):
            return c.replace("neu_", "")
    return "?"


def _gray(path, size):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.int16)


def dhash(path, size=16):
    """dHash ความละเอียดสูง (16x16 -> 256 bit) — ทนภาพ texture มากกว่า 8x8"""
    a = _gray(path, size)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def pixel_vec(path, size=32):
    """เวกเตอร์พิกเซล normalized (mean 0, ||.||=1) ไว้ยืนยันด้วย cosine similarity"""
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    v = np.asarray(im, dtype=np.float32).flatten()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())


def collect(split_dir):
    out = []
    img_dir = split_dir / "images"
    if not img_dir.is_dir():
        return out
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            try:
                out.append((p.name, class_of(p.name), dhash(p), pixel_vec(p)))
            except Exception as e:
                print(f"  ! อ่านไม่ได้: {p.name} ({e})")
    return out


def main():
    ap = argparse.ArgumentParser(description="ตรวจภาพรั่วข้าม split ด้วย perceptual hash")
    ap.add_argument("--data", default="merged_dataset_gray",
                    help="โฟลเดอร์ dataset ที่มี train/ valid/ test/")
    ap.add_argument("--threshold", type=int, default=24,
                    help="Hamming distance สูงสุดของ dHash 256-bit ที่ถือว่า 'ผู้ต้องสงสัย' (0=เหมือนเป๊ะ)")
    ap.add_argument("--cos", type=float, default=0.92,
                    help="cosine similarity ขั้นต่ำของเวกเตอร์พิกเซล 32x32 ที่ยืนยันว่าเป็นภาพเดียวกันจริง")
    ap.add_argument("--cross-class", action="store_true",
                    help="เทียบข้ามคลาสด้วย (ปริยายเทียบเฉพาะคลาสเดียวกัน — leakage จริงต้อง label เดิม)")
    ap.add_argument("--out", default=None, help="เขียนผลเป็น JSON (ไม่ระบุ = ไม่เขียน)")
    args = ap.parse_args()

    root = Path(args.data) if Path(args.data).is_absolute() else BASE / args.data
    splits = {s: collect(root / s) for s in ("train", "valid", "test")}
    for s, items in splits.items():
        print(f"{s:6s}: {len(items)} ภาพ")

    pairs_to_check = [("train", "valid"), ("train", "test"), ("valid", "test")]
    findings = []
    for a, b in pairs_to_check:
        na = 0
        for name_a, cls_a, dh_a, pv_a in splits[a]:
            for name_b, cls_b, dh_b, pv_b in splits[b]:
                if not args.cross_class and cls_a != cls_b:
                    continue
                d = hamming(dh_a, dh_b)
                if d <= args.threshold:
                    cos = float(np.dot(pv_a, pv_b))       # ยืนยันด้วยพิกเซลจริง
                    if cos >= args.cos:
                        findings.append({
                            "split_a": a, "image_a": name_a,
                            "split_b": b, "image_b": name_b,
                            "class": cls_a if cls_a == cls_b else f"{cls_a}/{cls_b}",
                            "dhash_dist": d, "cosine": round(cos, 4),
                        })
                        na += 1
        print(f"  {a:5s} ↔ {b:5s}: พบคู่ยืนยัน {na}")

    findings.sort(key=lambda f: (-f["cosine"], f["dhash_dist"]))
    print(f"\nรวมคู่ที่ยืนยันว่าเป็นภาพเดียวกัน/เกือบเหมือน ข้าม split: {len(findings)}")
    for f in findings[:40]:
        print(f"  [d{f['dhash_dist']:3d} cos{f['cosine']:.3f}] {f['class']:16s} "
              f"{f['split_a']}/{f['image_a'][:42]}  ==  {f['split_b']}/{f['image_b'][:42]}")
    if len(findings) > 40:
        print(f"  ... อีก {len(findings) - 40} คู่")

    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else BASE / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "data": str(root), "threshold": args.threshold,
            "split_sizes": {s: len(v) for s, v in splits.items()},
            "n_cross_split_pairs": len(findings),
            "pairs": findings,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nเขียน: {out}")

    if findings:
        print("\n⚠  พบภาพคร่อม split — ควร re-split ให้ภาพซ้ำอยู่ split เดียวกัน "
              "แล้วเทรน/วัดผลใหม่ ก่อนอ้างตัวเลขในเล่ม")
    else:
        print("\n✓ ไม่พบภาพรั่วข้าม split ที่เกณฑ์นี้")


if __name__ == "__main__":
    main()
