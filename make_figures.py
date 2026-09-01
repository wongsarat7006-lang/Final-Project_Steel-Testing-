"""
สร้างรูปประกอบรายงานลงโฟลเดอร์ figures/

  1. training_curves.png   mAP50 / loss เทียบ epoch ของแต่ละ run
  2. per_class_map.png      แท่ง mAP50 ต่อคลาส เทียบระหว่างโมเดล (จากไฟล์ eval JSON)
  3. class_distribution.png จำนวน instance ต่อคลาสใน merged_dataset (train)
  4. confusion_compare.png  confusion matrix ของแต่ละ run วางเทียบกัน (ถ้ามีไฟล์)

วิธีใช้:
    python make_figures.py
    python make_figures.py --runs train-clean train-balanced
    python make_figures.py --evals train-clean:evaluation_results.json train-balanced:eval_balanced.json
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager


def _set_thai_font():
    """เลือกฟอนต์ที่แสดงภาษาไทยได้ (Windows) ไม่งั้น matplotlib จะขึ้นกล่องสี่เหลี่ยม"""
    for name in ("Leelawadee UI", "Tahoma", "TH Sarabun New", "Angsana New"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


_set_thai_font()

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs" / "detect"
FIG_DIR = BASE_DIR / "figures"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
           "rolled-in_scale", "scratches", "rust", "crack"]


def read_results_csv(run):
    path = RUNS_DIR / run / "results.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    g = lambda k: [float(r[k].strip()) for r in rows]
    return {
        "epoch": g("epoch"),
        "mAP50": g("metrics/mAP50(B)"),
        "mAP50-95": g("metrics/mAP50-95(B)"),
        "train/box_loss": g("train/box_loss"),
        "val/box_loss": g("val/box_loss"),
    }


def fig_training_curves(runs):
    data = {r: read_results_csv(r) for r in runs}
    data = {k: v for k, v in data.items() if v}
    if not data:
        print("  ข้าม training_curves — ไม่พบ results.csv")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, d in data.items():
        ax[0].plot(d["epoch"], d["mAP50"], label=f"{name} mAP50")
        ax[0].plot(d["epoch"], d["mAP50-95"], "--", label=f"{name} mAP50-95")
        ax[1].plot(d["epoch"], d["train/box_loss"], label=f"{name} train")
        ax[1].plot(d["epoch"], d["val/box_loss"], "--", label=f"{name} val")
    ax[0].set(title="mAP เทียบ epoch (validation)", xlabel="epoch", ylabel="mAP")
    ax[1].set(title="box loss เทียบ epoch", xlabel="epoch", ylabel="loss")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=130)
    plt.close(fig)
    print("  เขียน figures/training_curves.png")


def _per_class_from_eval(path):
    """ดึง {class: mAP50} จากไฟล์ evaluation JSON (รองรับ key 'stage2' หรือ top-level)"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    node = obj.get("stage2", obj)
    pc = node.get("per_class")
    if not pc:
        return None
    return {r["class"]: r.get("mAP50", r.get("f1", 0.0)) for r in pc}


def fig_per_class_map(evals):
    series = {}
    for spec in evals:
        name, _, path = spec.partition(":")
        if not path or not Path(path).exists():
            print(f"  ข้าม eval '{spec}' — ไม่พบไฟล์")
            continue
        d = _per_class_from_eval(path)
        if d:
            series[name] = d
    if not series:
        print("  ข้าม per_class_map — ไม่มีไฟล์ eval ที่ใช้ได้")
        return
    import numpy as np
    x = np.arange(len(CLASSES))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (name, d) in enumerate(series.items()):
        vals = [d.get(c, 0.0) for c in CLASSES]
        ax.bar(x + i * w, vals, w, label=name)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set(title="mAP50 ต่อคลาส (test split)", ylabel="mAP50", ylim=(0, 1))
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_map.png", dpi=130)
    plt.close(fig)
    print("  เขียน figures/per_class_map.png")


def fig_class_distribution():
    lbl_dir = BASE_DIR / "merged_dataset" / "train" / "labels"
    if not lbl_dir.is_dir():
        print("  ข้าม class_distribution — ไม่พบ merged_dataset/train/labels")
        return
    inst = Counter()
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            p = line.split()
            if p:
                inst[int(p[0])] += 1
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = [inst[i] for i in range(len(CLASSES))]
    bars = ax.bar(CLASSES, vals, color="#4C72B0")
    ax.bar_label(bars)
    ax.set(title="จำนวน instance ต่อคลาส — merged_dataset/train", ylabel="instances")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=130)
    plt.close(fig)
    print("  เขียน figures/class_distribution.png")


def fig_confusion_compare(runs):
    imgs = []
    for r in runs:
        for cand in (RUNS_DIR / r / "confusion_matrix_normalized.png",
                     RUNS_DIR / f"val-{r}" / "confusion_matrix_normalized.png"):
            if cand.exists():
                imgs.append((r, cand)); break
    if len(imgs) < 1:
        print("  ข้าม confusion_compare — ไม่พบ confusion_matrix_normalized.png "
              "(รัน evaluate.py --mode stage2 ก่อน)")
        return
    fig, ax = plt.subplots(1, len(imgs), figsize=(7 * len(imgs), 6))
    if len(imgs) == 1:
        ax = [ax]
    for a, (name, path) in zip(ax, imgs):
        a.imshow(mpimg.imread(path))
        a.set_title(name); a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_compare.png", dpi=130)
    plt.close(fig)
    print("  เขียน figures/confusion_compare.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["train-clean", "train-balanced"])
    ap.add_argument("--evals", nargs="+",
                    default=["train-clean:evaluation_results_clean.json",
                             "train-balanced:evaluation_results_balanced.json"],
                    help="รายการ name:path.json ของผล eval Stage 2")
    args = ap.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    print(f"สร้างรูปลง {FIG_DIR}/")
    fig_training_curves(args.runs)
    fig_per_class_map(args.evals)
    fig_class_distribution()
    fig_confusion_compare(args.runs)
    print("เสร็จ")


if __name__ == "__main__":
    main()
