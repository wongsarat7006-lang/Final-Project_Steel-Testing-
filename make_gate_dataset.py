"""
make_gate_dataset.py — สร้างชุดเทรน classifier "พื้นผิวเหล็ก / ไม่ใช่" (Stage 0 gate)

    python make_gate_dataset.py

ผลลัพธ์: dataset_gate/{train,val}/{steel,not_steel}/  (โครงของ ultralytics classify)

steel  (positive)  = ภาพจาก merged_dataset (เฉพาะ neu_/rust_/real* — ตัด crack_ ทิ้ง
                     เพราะ crack_dataset เป็นรอยแตก "คอนกรีต" ไม่ใช่เหล็ก)
not_steel (neg)    = DTD (texture 47 หมวด) + Imagenette (วัตถุ/ฉาก/สัตว์)
                     + คอนกรีตจาก crack_dataset (โชว์ให้ gate แยกคอนกรีตออกจากเหล็ก)

ต้องมี downloads/gate_neg/dtd.tar.gz + downloads/gate_neg/imagenette.tgz (ดู DATA_COLLECTION / NEXT_STEPS)
สคริปต์จะแตกไฟล์เองถ้ายังไม่ได้แตก
"""
import random
import shutil
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
NEG_SRC = BASE / "downloads" / "gate_neg"
OUT = BASE / "dataset_gate"
SEED = 0
VAL_FRAC = 0.10
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _extract(archive: Path, marker: Path):
    if marker.exists():
        return
    if not archive.exists():
        raise SystemExit(f"ไม่พบ {archive} — โหลดก่อน (ดู NEXT_STEPS)")
    print(f"แตกไฟล์ {archive.name} ...")
    with tarfile.open(archive) as t:
        t.extractall(NEG_SRC)


def _imgs(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in EXTS and p.is_file()]


def _copy(files, dst: Path, prefix: str):
    dst.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(files):
        shutil.copy(p, dst / f"{prefix}_{i:05d}{p.suffix.lower()}")


def main():
    random.seed(SEED)
    _extract(NEG_SRC / "dtd.tar.gz", NEG_SRC / "dtd" / "images")
    _extract(NEG_SRC / "imagenette.tgz", NEG_SRC / "imagenette2-160")

    # ---------- positives : merged_dataset (ตัด crack_ ทิ้ง) ----------
    def steel_imgs(split):
        d = BASE / "merged_dataset" / split / "images"
        return [p for p in d.iterdir()
                if p.suffix.lower() in EXTS and not p.name.startswith("crack_")]

    steel_train, steel_val = steel_imgs("train"), steel_imgs("valid")
    random.shuffle(steel_train)
    n_pos = len(steel_train)
    print(f"steel: train {n_pos} / val {len(steel_val)}")

    # ---------- negatives ----------
    dtd = _imgs(NEG_SRC / "dtd" / "images")
    inet = _imgs(NEG_SRC / "imagenette2-160")
    concrete = [p for p in (BASE / "crack_dataset" / "train" / "images").iterdir()
                if p.suffix.lower() in EXTS]
    random.shuffle(dtd); random.shuffle(inet); random.shuffle(concrete)

    # สัดส่วน neg ~ เท่ากับ pos : DTD เยอะสุด, imagenette รอง, concrete พอเป็นตัวแทน
    n_concrete = min(500, len(concrete))
    n_inet = min(1500, len(inet))
    n_dtd = max(0, (n_pos + len(steel_val)) - n_concrete - n_inet)
    n_dtd = min(n_dtd, len(dtd)) or len(dtd)
    neg = ([("dtd", p) for p in dtd[:n_dtd]]
           + [("inet", p) for p in inet[:n_inet]]
           + [("concrete", p) for p in concrete[:n_concrete]])
    random.shuffle(neg)
    cut = int(len(neg) * (1 - VAL_FRAC))
    neg_train, neg_val = neg[:cut], neg[cut:]
    print(f"not_steel: train {len(neg_train)} / val {len(neg_val)} "
          f"(dtd {n_dtd}, imagenette {n_inet}, concrete {n_concrete})")

    # ---------- write ----------
    if OUT.exists():
        shutil.rmtree(OUT)
    _copy(steel_train, OUT / "train" / "steel", "steel")
    _copy(steel_val, OUT / "val" / "steel", "steel")
    _copy([p for _, p in neg_train], OUT / "train" / "not_steel", "neg")
    _copy([p for _, p in neg_val], OUT / "val" / "not_steel", "neg")

    for split in ("train", "val"):
        for cls in ("steel", "not_steel"):
            n = len(list((OUT / split / cls).iterdir()))
            print(f"  {split}/{cls}: {n}")
    print(f"\nเสร็จ -> {OUT}\nต่อด้วย:  python train_gate.py")


if __name__ == "__main__":
    main()
