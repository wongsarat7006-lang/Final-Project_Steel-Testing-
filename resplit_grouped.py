"""
Group-aware stratified re-split — แก้ data leakage จากภาพถ่ายรัว/ซ้ำที่ถูกสุ่มแยกคนละ split

ต่างจาก resplit_dataset.py: จับกลุ่มภาพที่ "เกือบเหมือนกัน" (near-duplicate) ด้วย
perceptual hash + cosine ของพิกเซล แล้วบังคับให้ทั้งกลุ่มอยู่ split เดียวกัน
=> valid/test ไม่มีภาพที่โมเดลเคยเห็นตอนเทรน

ใช้กับ merged_dataset/ และ merged_dataset_gray/ พร้อมกัน (ชื่อไฟล์ตรงกัน => split เดียวกัน)
คำนวณ assignment จาก dataset แรก แล้ว apply เหมือนกันทุก dataset

    python resplit_grouped.py --dry-run
    python resplit_grouped.py --val 0.15 --test 0.15 --seed 0
    python check_leakage.py --data merged_dataset_gray     # ยืนยันหลังรัน: ควรได้ 0 คู่

หลังรัน: python make_oversampled_list.py --dataset merged_dataset_gray ; python train.py ...
"""
import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _feats(path, hsize=16, psize=32):
    im = Image.open(path).convert("L")
    g = np.asarray(im.resize((hsize + 1, hsize), Image.LANCZOS), dtype=np.int16)
    dh = np.packbits((g[:, 1:] > g[:, :-1]).flatten())          # 256-bit dHash
    v = np.asarray(im.resize((psize, psize), Image.LANCZOS), dtype=np.float32).flatten()
    v -= v.mean()
    n = np.linalg.norm(v)
    v = v / n if n > 1e-6 else v
    return dh, v


def gather(dataset):
    """คืน list ของ (stem, image_path, label_path, primary_class)"""
    items = []
    freq = Counter()
    for split in SPLITS:
        lbl_dir = dataset / split / "labels"
        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if p:
                        freq[int(p[0])] += 1
    freq = defaultdict(lambda: 1, freq)

    for split in SPLITS:
        img_dir = dataset / split / "images"
        lbl_dir = dataset / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            classes = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if p:
                        classes.append(int(p[0]))
            primary = min(classes, key=lambda c: freq[c]) if classes else -1
            items.append((img.stem, img, lbl, primary))
    return items


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_groups(items, cos_thr, dh_thr):
    """union-find บนคู่ภาพ near-duplicate -> คืน list ของ group_id ต่อ item"""
    n = len(items)
    print(f"  คำนวณ features ของ {n} ภาพ...")
    dhs = np.zeros((n, 32), dtype=np.uint8)
    pv = np.zeros((n, 32 * 32), dtype=np.float32)
    for i, (_stem, img, _lbl, _c) in enumerate(items):
        dhs[i], pv[i] = _feats(img)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n}")

    print("  หา near-duplicate (cosine matrix)...")
    uf = UF(n)
    n_edges = 0
    _POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)
    block = 512
    for s in range(0, n, block):
        e = min(n, s + block)
        sim = pv[s:e] @ pv.T                       # (block, n) cosine
        for local, i in enumerate(range(s, e)):
            cand = np.where(sim[local] >= cos_thr)[0]
            cand = cand[cand > i]
            for j in cand:
                hd = int(_POP[dhs[i] ^ dhs[j]].sum())
                if hd <= dh_thr:
                    uf.union(i, int(j))
                    n_edges += 1
    print(f"  near-duplicate edges: {n_edges}")

    gid = [uf.find(i) for i in range(n)]
    # remap เป็นเลขต่อเนื่อง
    remap = {g: k for k, g in enumerate(sorted(set(gid)))}
    return [remap[g] for g in gid]


def main():
    ap = argparse.ArgumentParser(description="Group-aware stratified re-split")
    ap.add_argument("--datasets", nargs="+",
                    default=["merged_dataset_gray", "merged_dataset"],
                    help="โฟลเดอร์ dataset (คำนวณ assignment จากตัวแรก apply เหมือนกันทุกตัว)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cos", type=float, default=0.93, help="cosine ขั้นต่ำที่ถือว่าใกล้กัน")
    ap.add_argument("--dhash", type=int, default=32, help="Hamming สูงสุดของ dHash 256-bit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    datasets = [Path(d) if Path(d).is_absolute() else BASE / d for d in args.datasets]
    for d in datasets:
        if not d.exists():
            raise SystemExit(f"ไม่พบ {d}")

    ref = datasets[0]
    print(f"คำนวณ assignment จาก: {ref.name}")
    items = gather(ref)
    print(f"  รวม {len(items)} ภาพจาก {len(SPLITS)} split เดิม")

    groups = build_groups(items, args.cos, args.dhash)
    n_groups = len(set(groups))
    gsize = Counter(groups)
    multi = sum(1 for v in gsize.values() if v > 1)
    print(f"  => {n_groups} กลุ่ม ({multi} กลุ่มมีสมาชิก >1, กลุ่มใหญ่สุด {max(gsize.values())} ภาพ)")

    # ---- stratified assignment ของ "กลุ่ม" (ไม่ใช่ภาพ) ----
    # primary class ของกลุ่ม = คลาสหายากสุดในกลุ่ม
    import random
    freq_all = Counter(it[3] for it in items)
    freq_all = defaultdict(lambda: 1, freq_all)
    group_members = defaultdict(list)
    for idx, g in enumerate(groups):
        group_members[g].append(idx)
    group_primary = {}
    for g, idxs in group_members.items():
        cls = [items[i][3] for i in idxs]
        group_primary[g] = min(cls, key=lambda c: freq_all[c])

    by_class = defaultdict(list)
    for g, idxs in group_members.items():
        by_class[group_primary[g]].append((g, len(idxs)))

    rng = random.Random(args.seed)
    assign = {}  # stem -> split
    for cls, glist in by_class.items():
        rng.shuffle(glist)
        total = sum(sz for _g, sz in glist)
        want_test = total * args.test
        want_val = total * args.val
        acc_test = acc_val = 0
        for g, sz in glist:
            if acc_test < want_test:
                s = "test"; acc_test += sz
            elif acc_val < want_val:
                s = "valid"; acc_val += sz
            else:
                s = "train"
            for i in group_members[g]:
                assign[items[i][0]] = s

    # ---- สรุป ----
    counts = Counter()
    dist = {s: Counter() for s in SPLITS}
    for stem, _img, lbl, _c in items:
        s = assign[stem]
        counts[s] += 1
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if p:
                    dist[s][int(p[0])] += 1
    print("\nจำนวนภาพต่อ split:", dict(counts))
    for s in SPLITS:
        print(f"  {s:6} class dist: {dict(sorted(dist[s].items()))}")

    if args.dry_run:
        print("\n[dry-run] ไม่ย้ายไฟล์")
        return

    for dataset in datasets:
        print(f"\nย้ายไฟล์: {dataset.name}")
        d_items = gather(dataset)
        staging = dataset / "_restaged"
        if staging.exists():
            shutil.rmtree(staging)
        for s in SPLITS:
            (staging / s / "images").mkdir(parents=True, exist_ok=True)
            (staging / s / "labels").mkdir(parents=True, exist_ok=True)
        missing = 0
        for stem, img, lbl, _c in d_items:
            s = assign.get(stem)
            if s is None:
                missing += 1
                s = "train"
            shutil.copy(img, staging / s / "images" / img.name)
            if lbl.exists():
                shutil.copy(lbl, staging / s / "labels" / lbl.name)
            else:
                (staging / s / "labels" / f"{stem}.txt").touch()
        if missing:
            print(f"  ! {missing} ภาพไม่มีใน assignment (ใส่ train)")
        for s in SPLITS:
            for sub in ("images", "labels"):
                old = dataset / s / sub
                if old.exists():
                    shutil.rmtree(old)
                shutil.move(str(staging / s / sub), str(old))
            for c in (dataset / s).glob("labels.cache"):
                c.unlink()
        shutil.rmtree(staging)
        print(f"  เสร็จ: {dataset.name}")

    print("\nต่อ: python make_oversampled_list.py --dataset merged_dataset_gray")
    print("     python check_leakage.py --data merged_dataset_gray   (ยืนยันควรได้ 0)")


if __name__ == "__main__":
    main()
