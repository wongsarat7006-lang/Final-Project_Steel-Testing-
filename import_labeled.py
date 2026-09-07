"""
import_labeled.py — ดึง dataset ที่มี label อยู่แล้ว (Roboflow / Kaggle YOLO export)
เข้ามาเป็น round ของ active-learning loop โดย remap ชื่อคลาสให้ตรง 8 คลาสของเรา

    python import_labeled.py --src downloads\corrosion-yolov11 --round 1 \
        --map "corrosion=rust, rust=rust, crack=crack, Crack=crack"

- อ่าน data.yaml ของชุดต้นทาง (names: [...]) เพื่อรู้ว่า class id ไหน = ชื่ออะไร
- เก็บเฉพาะกล่องที่ชื่อคลาสอยู่ใน --map แล้วเขียน class id ใหม่เป็นของเรา
- รองรับทั้งแบบมี split (train/valid/test) และแบบมี images/ labels/ ตรง ๆ
- ปลายทาง: dataset_real/round<N>/{images,labels,classes.txt,data.yaml}
  -> ต่อด้วย  python run_round.py train --round <N> --skip-label-check
     (label เป็นของคนอยู่แล้ว แค่ควร spot-check กรอบ + ชนิดก่อนเทรน)

ชื่อคลาสของเรา: crazing inclusion patches pitted_surface rolled-in_scale scratches rust crack
(ชุดจากเน็ตส่วนใหญ่ครอบแค่ rust/crack — อีก 4 คลาสผิวโรงงานมีใน NEU/GC10 อยู่แล้ว)
"""
import argparse
import shutil
from pathlib import Path

import pipeline as P

BASE = Path(__file__).resolve().parent
CLASSES = P.DEFECT_CLASSES
NAME_TO_ID = {c.lower(): i for i, c in enumerate(CLASSES)}


def parse_map(s):
    """'corrosion=rust, crack=crack' -> {'corrosion': 7-ish id, ...} (key = lower)"""
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"--map ต้องเป็น src=our รูปแบบ ไม่ใช่ '{pair}'")
        src, ours = (x.strip() for x in pair.split("=", 1))
        if ours.lower() not in NAME_TO_ID:
            raise SystemExit(f"'{ours}' ไม่ใช่คลาสของเรา ({', '.join(CLASSES)})")
        out[src.lower()] = NAME_TO_ID[ours.lower()]
    if not out:
        raise SystemExit("--map ว่าง")
    return out


def read_src_names(src):
    """คืน dict {class_id -> name} จาก data.yaml ของชุดต้นทาง"""
    yml = next((p for p in [src / "data.yaml", src / "data.yml"] if p.exists()), None)
    if not yml:
        raise SystemExit(f"ไม่พบ data.yaml ใน {src} — ชี้ --src ไปโฟลเดอร์ที่มี data.yaml")
    names = None
    txt = yml.read_text(encoding="utf-8")
    # รองรับทั้ง  names: ['a','b']  และ  names:\n  0: a\n  1: b
    for line in txt.splitlines():
        if line.strip().startswith("names:") and "[" in line:
            inside = line.split("[", 1)[1].rsplit("]", 1)[0]
            names = [x.strip().strip("'\"") for x in inside.split(",") if x.strip()]
            break
    if names is None:
        names, collecting = [], False
        for line in txt.splitlines():
            if line.strip().startswith("names:"):
                collecting = True
                continue
            if collecting:
                s = line.strip()
                if not s or ":" not in s and not s.startswith("-"):
                    break
                if s.startswith("-"):
                    names.append(s[1:].strip().strip("'\""))
                else:  # "0: name"
                    names.append(s.split(":", 1)[1].strip().strip("'\""))
    if not names:
        raise SystemExit(f"อ่าน names จาก {yml} ไม่ได้")
    return {i: n for i, n in enumerate(names)}


def iter_split_dirs(src):
    """คืน [(images_dir, labels_dir), ...] — รองรับ split หรือ flat"""
    pairs = []
    for sub in ["train", "valid", "val", "test", ""]:
        d = src / sub if sub else src
        img = d / "images"
        lbl = d / "labels"
        if img.is_dir() and lbl.is_dir():
            pairs.append((img, lbl))
    if not pairs:
        raise SystemExit(f"ไม่พบ images/ + labels/ ใน {src} (หรือใน train/valid/test)")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="โฟลเดอร์ dataset ที่โหลดมา (มี data.yaml)")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--map", required=True,
                    help="remap ชื่อคลาส เช่น \"corrosion=rust, crack=crack\"")
    ap.add_argument("--keep-empty", action="store_true",
                    help="เก็บภาพที่ไม่เหลือกล่องเลยไว้เป็น negative (default: ข้าม)")
    ap.add_argument("--max-box-frac", type=float, default=1.0,
                    help="ทิ้งกรอบที่พื้นที่ > สัดส่วนนี้ของภาพ (กรอบ polygon ที่ระเบิดใหญ่ "
                         "จาก aug หมุน) เช่น 0.45")
    ap.add_argument("--prefix", default=None, help="prefix ชื่อไฟล์ (default: rb<N>_)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์ {src}")
    cmap = parse_map(args.map)
    src_names = read_src_names(src)
    prefix = args.prefix or f"rb{args.round}_"

    # src class id -> our class id
    id_remap = {}
    for sid, sname in src_names.items():
        if sname.lower() in cmap:
            id_remap[sid] = cmap[sname.lower()]
    if not id_remap:
        raise SystemExit(
            f"ไม่มีคลาสใน --map ตรงกับ names ของชุดต้นทาง: {list(src_names.values())}")
    print("remap:", {src_names[k]: CLASSES[v] for k, v in id_remap.items()})

    out = BASE / "dataset_real" / f"round{args.round}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    n_img = n_box = n_skip_empty = 0
    per_class = {c: 0 for c in CLASSES}
    for img_dir, lbl_dir in iter_split_dirs(src):
        for ip in sorted(img_dir.iterdir()):
            if ip.suffix.lower() not in P.IMAGE_EXTS:
                continue
            lp = lbl_dir / f"{ip.stem}.txt"
            kept = []
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    try:
                        sid = int(float(parts[0]))
                        vals = [float(v) for v in parts[1:]]
                    except ValueError:
                        continue  # ข้ามบรรทัด README ที่หลุดเข้ามา
                    if sid not in id_remap:
                        continue
                    if len(vals) == 4:                       # bbox: cx cy w h
                        cx, cy, bw, bh = vals
                    elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon (seg) -> bbox
                        xs, ys = vals[0::2], vals[1::2]
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    else:
                        continue
                    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                    bw, bh = min(bw, 1.0), min(bh, 1.0)
                    if bw <= 1e-3 or bh <= 1e-3:
                        continue
                    if bw * bh > args.max_box_frac:
                        continue
                    oid = id_remap[sid]
                    kept.append(f"{oid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    per_class[CLASSES[oid]] += 1
            if not kept and not args.keep_empty:
                n_skip_empty += 1
                continue
            stem = f"{prefix}{ip.stem}"
            shutil.copy(ip, out / "images" / f"{stem}{ip.suffix.lower()}")
            body = ("\n".join(kept) + "\n") if kept else ""
            (out / "labels" / f"{stem}.txt").write_text(body, encoding="utf-8")
            n_img += 1
            n_box += len(kept)

    (out / "classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")

    print(f"\nนำเข้า {n_img} ภาพ / {n_box} กรอบ -> {out}")
    print("ต่อคลาส:", {k: v for k, v in per_class.items() if v})
    if n_skip_empty:
        print(f"ข้าม {n_skip_empty} ภาพที่ไม่เหลือกรอบ (ใส่ --keep-empty ถ้าอยากเก็บเป็น negative)")
    print(f"""
ขั้นต่อไป: เปิด {out}/ ใน LabelImg ตรวจกรอบ/ชนิดผ่าน ๆ (label เป็นของคนต้นทาง มักโอเค)
แล้ว:  python run_round.py train --round {args.round} --skip-label-check""")


if __name__ == "__main__":
    main()
