"""
รวมผล evaluate.py --mode stage2 จากหลาย seed -> mean ± std ต่อคลาส (confidence interval)

    python aggregate_seeds.py results/stage2_seed0.json results/stage2_seed1.json results/stage2_seed2.json
    python aggregate_seeds.py --glob "results/stage2_grayn2_seed*.json" --out results/stage2_multiseed.json

ใช้ประกอบบท "ผลการทดลอง" — รายงาน mAP50 เป็น mean ± std (n=3) แทนตัวเลขเดียว
"""
import argparse
import glob as globmod
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _overall(d):
    return d.get("stage2", d).get("overall", d.get("overall", {}))


def _per_class(d):
    return d.get("stage2", d).get("per_class", d.get("per_class", []))


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v)


def main():
    ap = argparse.ArgumentParser(description="รวมผลหลาย seed เป็น mean ± std")
    ap.add_argument("files", nargs="*", help="ไฟล์ results/stage2_*.json")
    ap.add_argument("--glob", default=None, help="pattern แทนการระบุไฟล์ทีละตัว")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = list(args.files)
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    paths = [p for p in dict.fromkeys(paths)]
    if len(paths) < 2:
        raise SystemExit("ต้องมีอย่างน้อย 2 ไฟล์ (2 seed)")

    dicts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    print(f"รวม {len(dicts)} seed:")
    for p in paths:
        print("  ", p)

    metrics = ["mAP50", "mAP50-95", "precision", "recall"]
    overall = {}
    for k in metrics:
        vals = [_overall(d).get(k) for d in dicts if _overall(d).get(k) is not None]
        m, s = mean_std(vals)
        overall[k] = {"mean": round(m, 4), "std": round(s, 4), "n": len(vals), "values": [round(v, 4) for v in vals]}

    # per-class: จับคู่ด้วยชื่อคลาส
    classes = [c["class"] for c in _per_class(dicts[0])]
    per_class = {}
    for cls in classes:
        row = {}
        for k in ["mAP50", "mAP50-95", "precision", "recall"]:
            vals = []
            for d in dicts:
                for c in _per_class(d):
                    if c["class"] == cls and c.get(k) is not None:
                        vals.append(c[k])
            m, s = mean_std(vals)
            row[k] = {"mean": round(m, 4), "std": round(s, 4)}
        per_class[cls] = row

    print(f"\n{'metric':<12}{'mean':>10}{'std':>9}   values")
    for k in metrics:
        o = overall[k]
        print(f"{k:<12}{o['mean']:>10.4f}{o['std']:>9.4f}   {o['values']}")
    print(f"\n{'class':<16}{'mAP50 mean':>12}{'±std':>9}{'recall mean':>13}{'±std':>9}")
    for cls, r in per_class.items():
        print(f"{cls:<16}{r['mAP50']['mean']:>12.4f}{r['mAP50']['std']:>9.4f}"
              f"{r['recall']['mean']:>13.4f}{r['recall']['std']:>9.4f}")

    out = {"n_seeds": len(dicts), "sources": paths, "overall": overall, "per_class": per_class}
    dest = args.out or (BASE / "results" / "stage2_multiseed.json")
    Path(dest).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nเขียน: {dest}")


if __name__ == "__main__":
    main()
