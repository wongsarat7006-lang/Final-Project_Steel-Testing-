"""
Smoke test — รันเร็ว ๆ กัน regression ก่อน commit / ก่อนส่งงาน

    python test_smoke.py

เช็ค:
  - ทุกโมดูล import ได้
  - build_regions / cross_region_nms / load_class_conf ทำงานถูก (ไม่ต้องมีโมเดล)
  - ถ้ามี DMS46_v1.pt + best.pt + test_images/ : รัน pipeline จริง 1 ภาพ แล้วเช็คโครง output

exit 0 = ผ่านหมด, exit 1 = มีอย่างน้อย 1 ข้อ fail
"""
import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
_fail = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        _fail.append(name)
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


def t_imports():
    import pipeline, evaluate, evaluate_real, app, tune_thresholds  # noqa: F401
    import fix_labels, make_grayscale_dataset, make_oversampled_list  # noqa: F401


def t_build_regions():
    import numpy as np
    import pipeline as P
    mask = np.zeros((100, 200), np.uint8)
    boxes, meta = P.build_regions(mask, (100, 200))          # ไม่มีเหล็กเลย
    assert meta["fallback_full_image"] and boxes == [(0, 0, 200, 100)], (boxes, meta)
    mask[10:90, 20:180] = 255                                 # เหล็กเต็ม ๆ
    boxes, meta = P.build_regions(mask, (100, 200))
    assert not meta["fallback_full_image"] and meta["metal_found"], meta
    assert len(boxes) >= 1


def t_nms():
    import pipeline as P
    d = lambda c, s, box: {"class": c, "confidence": s, "bbox_xyxy_global": box}
    dets = [d("rust", 0.9, [0, 0, 10, 10]), d("rust", 0.6, [1, 1, 11, 11]),  # ซ้ำ
            d("rust", 0.8, [50, 50, 60, 60]), d("crack", 0.7, [0, 0, 10, 10])]
    kept = P.cross_region_nms(dets, iou_thresh=0.5)
    assert len(kept) == 3, [k["class"] + str(k["confidence"]) for k in kept]


def t_class_conf_optional():
    import pipeline as P
    # ไม่มีไฟล์ -> None (ไม่ error)
    got = P.load_class_conf(BASE / "____nope____.json")
    assert got is None


def t_pipeline_end_to_end():
    import pipeline as P
    need = [P.STAGE1_MODEL_PATH, P.STAGE2_MODEL_PATH]
    imgs = [p for p in (BASE / "test_images").glob("*") if p.suffix.lower() in P.IMAGE_EXTS]
    if not all(p.exists() for p in need) or not imgs:
        print("       (ข้าม — ไม่มีโมเดลหรือ test_images/)")
        return
    dev = P.resolve_device("auto")
    s1, s2 = P.load_models(dev)
    out_dir = BASE / "pipeline_results" / "_smoke"
    summ = P.process_image(imgs[0], s1, s2, out_dir, 0.4, dev)
    assert summ and "regions" in summ and "fallback_full_image" in summ
    assert (out_dir / f"{imgs[0].stem}_result.json").exists()
    assert (out_dir / f"{imgs[0].stem}_result.jpg").exists()


for n, f in [
    ("imports", t_imports),
    ("build_regions", t_build_regions),
    ("cross_region_nms", t_nms),
    ("load_class_conf optional", t_class_conf_optional),
    ("pipeline end-to-end", t_pipeline_end_to_end),
]:
    check(n, f)

print()
if _fail:
    print(f"FAILED {len(_fail)}: {', '.join(_fail)}")
    sys.exit(1)
print("ผ่านหมด")
