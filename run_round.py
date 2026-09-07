"""
run_round.py — วนรอบ active-learning หนึ่งรอบด้วย "คำสั่งเดียว"

เป้าหมาย: ทำให้ loop ใน DATA_COLLECTION.md เป็น 2 คำสั่ง (มีขั้น label ของคนคั่นกลาง)

    # 1. มีโฟลเดอร์ภาพถ่ายจริงชุดใหม่แล้ว -> ให้โมเดลร่าง label
    python run_round.py draft --photos photos_round1/ --round 1

    # 2. คนเปิด dataset_real/round1/ ใน LabelImg แก้กรอบให้ถูก (ขั้นนี้คนทำ)

    # 3. merge + retrain + eval ทั้งชุดในคำสั่งเดียว
    python run_round.py train --round 1

    # ถ้ารอบนั้นผลแย่ลง -> ถอน merge ออก กลับไปสภาพก่อนหน้า
    python run_round.py rollback --round 1

รอบถัดไป: `draft --round 2` จะหยิบ weights ของ train-real1 มาเป็นตัวร่างอัตโนมัติ
(ร่างแม่นขึ้น คนแก้น้อยลง) — ดู DATA_COLLECTION.md สำหรับภาพรวม
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = sys.executable
DATASET = BASE / "merged_dataset"          # ชุดเทรนหลัก (RGB) — track ภาพจริง/โปรดักต์
REAL_ROOT = BASE / "dataset_real"          # ที่เก็บ label ร่าง/แก้แล้ว ต่อรอบ
RESULTS = BASE / "results"


def run(cmd, **kw):
    print("\n$ " + " ".join(str(c) for c in cmd) + "\n")
    r = subprocess.run([str(c) for c in cmd], cwd=BASE, **kw)
    if r.returncode != 0:
        raise SystemExit(f"คำสั่งล้มเหลว (exit {r.returncode}) — หยุด")
    return r


def prev_weights(rnd):
    """weights ของรอบก่อนหน้า (train-real<N-1>) ถ้ามี ไม่งั้น None"""
    for n in range(rnd - 1, 0, -1):
        w = BASE / "runs" / "detect" / f"train-real{n}" / "weights" / "best.pt"
        if w.exists():
            return w
    return None


# ---------------------------------------------------------------- draft
def cmd_draft(args):
    photos = Path(args.photos)
    if not photos.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์ภาพ {photos}")
    out = REAL_ROOT / f"round{args.round}"
    if (out / "labels").exists() and any((out / "labels").iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} มี label อยู่แล้ว — ใส่ --overwrite ถ้าจะร่างทับ")

    cmd = [PY, "annotate_bootstrap.py", "--src", photos, "--out", out, "--conf", args.conf]
    w = args.weights or prev_weights(args.round)
    if w:
        cmd += ["--weights", w]
        print(f"ใช้ weights รอบก่อน: {w}")
    else:
        print("ใช้ weights ปัจจุบันใน pipeline.py (ยังไม่มีรอบก่อนหน้า)")
    if not args.stage1:
        cmd += ["--no-stage1"]
    run(cmd)

    # เก็บสำเนา "ร่าง" ไว้เทียบตอน train (จับว่าคนแก้ label แล้วจริง)
    draft_copy = out / "labels_draft"
    if draft_copy.exists():
        shutil.rmtree(draft_copy)
    shutil.copytree(out / "labels", draft_copy)

    print(f"""
--------------------------------------------------------------------
ขั้นต่อไป (คนทำ): เปิด {out}/ ใน LabelImg / Label Studio
  - ลบกรอบเกิน / ขยับกรอบเพี้ยน / เพิ่มกรอบที่พลาด / แก้ชนิด
  - เก็บ format YOLO เดิม (classes.txt ให้มาแล้ว)
เสร็จแล้ว:  python run_round.py train --round {args.round}
--------------------------------------------------------------------""")


# ---------------------------------------------------------------- train
def cmd_train(args):
    rnd = args.round
    src = REAL_ROOT / f"round{rnd}"
    img_dir, lbl_dir = src / "images", src / "labels"
    if not lbl_dir.is_dir() or not any(lbl_dir.iterdir()):
        raise SystemExit(f"ไม่พบ label ใน {lbl_dir} — รัน draft ก่อน")

    # กันพลาด: เตือนถ้า label ยังเท่ากับร่าง (คนยังไม่แก้)
    draft_copy = src / "labels_draft"
    if draft_copy.is_dir() and not args.skip_label_check:
        same = True
        a = sorted(p.name for p in lbl_dir.glob("*.txt"))
        b = sorted(p.name for p in draft_copy.glob("*.txt"))
        if a != b:
            same = False
        else:
            for n in a:
                if (lbl_dir / n).read_text(encoding="utf-8").strip() != \
                   (draft_copy / n).read_text(encoding="utf-8").strip():
                    same = False
                    break
        if same:
            raise SystemExit(
                "label เหมือนตอนร่างเป๊ะ — ยังไม่ได้แก้มือ?\n"
                "ถ้าตั้งใจจะเทรนด้วยร่างดิบ ใส่ --skip-label-check")

    # --- merge เข้า train split (พร้อม manifest ให้ถอนกลับได้) ---
    dst_img = DATASET / "train" / "images"
    dst_lbl = DATASET / "train" / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    manifest_path = src / "merge_manifest.json"
    if manifest_path.exists() and not args.remerge:
        print(f"merge รอบนี้ทำไปแล้ว (มี {manifest_path.name}) — ข้ามขั้น merge "
              "(ใส่ --remerge ถ้าจะทำใหม่)")
        merged = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    else:
        merged = []
        for ip in sorted(img_dir.iterdir()):
            if not ip.is_file():
                continue
            lp = lbl_dir / f"{ip.stem}.txt"
            if not lp.exists():
                continue
            name_img = f"real{rnd}_{ip.name}"
            name_lbl = f"real{rnd}_{ip.stem}.txt"
            shutil.copy(ip, dst_img / name_img)
            shutil.copy(lp, dst_lbl / name_lbl)
            merged += [f"train/images/{name_img}", f"train/labels/{name_lbl}"]
        manifest_path.write_text(json.dumps(
            {"round": rnd, "when": datetime.now().isoformat(timespec="seconds"),
             "count_images": len(merged) // 2, "files": merged}, indent=2), encoding="utf-8")
        print(f"merge {len(merged)//2} ภาพเข้า {DATASET.name}/train  "
              f"(manifest: {manifest_path})")

    # --- oversample list + train + eval ---
    run([PY, "make_oversampled_list.py", "--dataset", DATASET.name])

    name = args.name or f"train-real{rnd}"
    warm = prev_weights(rnd)  # fine-tune ต่อจากรอบก่อน ถ้ามี
    base_model = args.model or (str(warm) if warm else "yolo11n.pt")
    run([PY, "train.py", "--recipe", "camera",
         "--data", f"{DATASET.name}/data_oversampled.yaml",
         "--model", base_model, "--name", name,
         "--epochs", args.epochs, "--batch", args.batch,
         "--patience", "40", "--device", args.device])

    best = BASE / "runs" / "detect" / name / "weights" / "best.pt"
    RESULTS.mkdir(exist_ok=True)
    out_json = RESULTS / f"stage2_real{rnd}.json"
    run([PY, "evaluate.py", "--mode", "stage2", "--weights", best,
         "--data", f"{DATASET.name}/data.yaml", "--out", out_json])

    # eval บนภาพจริงที่ hold out ไว้ (ถ้ามี) — สัญญาณตรงกับ use case
    if (BASE / "real_test" / "images").is_dir() and \
       any((BASE / "real_test" / "images").iterdir()):
        print("\nรัน evaluate_real.py (ภาพจริง hold-out) ...")
        subprocess.run([PY, "evaluate_real.py", "--weights", str(best),
                        "--out", str(RESULTS / f"real_after_round{rnd}.json")], cwd=BASE)

    print(f"""
--------------------------------------------------------------------
รอบ {rnd} เสร็จ
  weights : {best}
  metrics : {out_json}
เทียบกับรอบก่อน / results/stage2_train-gray-n2.json ว่า recall ภาพจริงขึ้นไหม
ดีขึ้น -> ชี้ pipeline.py STAGE2_MODEL_PATH มาที่ weights นี้ แล้วเริ่มรอบถัดไป:
    python run_round.py draft --photos photos_round{rnd+1}/ --round {rnd+1}
แย่ลง -> python run_round.py rollback --round {rnd}
--------------------------------------------------------------------""")


# ---------------------------------------------------------------- rollback
def cmd_rollback(args):
    src = REAL_ROOT / f"round{args.round}"
    manifest_path = src / "merge_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"ไม่พบ {manifest_path} — รอบนี้ยังไม่ถูก merge")
    files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    n = 0
    for rel in files:
        p = DATASET / rel
        if p.exists():
            p.unlink()
            n += 1
    manifest_path.rename(manifest_path.with_suffix(".json.undone"))
    run([PY, "make_oversampled_list.py", "--dataset", DATASET.name])
    print(f"ถอน {n} ไฟล์ของรอบ {args.round} ออกจาก {DATASET.name}/train แล้ว "
          f"(manifest -> .json.undone). เทรนใหม่ได้ตามต้องการ")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draft", help="ให้โมเดลร่าง label ให้ภาพชุดใหม่")
    d.add_argument("--photos", required=True, help="โฟลเดอร์ภาพถ่ายจริงชุดใหม่")
    d.add_argument("--round", type=int, required=True)
    d.add_argument("--conf", default="0.12", help="conf ต่ำ = ดึงกรอบมาเยอะไว้ก่อน")
    d.add_argument("--weights", default=None, help="บังคับใช้ weights นี้ร่าง (default: auto)")
    d.add_argument("--stage1", action="store_true",
                   help="ใช้ Stage 1 ด้วย (default: ข้าม — ภาพลูกค้าเล็งชิ้นเดียว)")
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(func=cmd_draft)

    t = sub.add_parser("train", help="merge + retrain + eval รอบนี้")
    t.add_argument("--round", type=int, required=True)
    t.add_argument("--epochs", default="120")
    t.add_argument("--batch", default="8")
    t.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "0"])
    t.add_argument("--model", default=None, help="base .pt (default: weights รอบก่อน ไม่งั้น yolo11n.pt)")
    t.add_argument("--name", default=None, help="ชื่อ run (default: train-real<N>)")
    t.add_argument("--skip-label-check", action="store_true",
                   help="ข้ามการเตือนว่า label ยังไม่ถูกแก้มือ")
    t.add_argument("--remerge", action="store_true", help="merge เข้า train ซ้ำแม้ทำไปแล้ว")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("rollback", help="ถอน merge ของรอบนี้ออกจาก train split")
    r.add_argument("--round", type=int, required=True)
    r.set_defaults(func=cmd_rollback)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
