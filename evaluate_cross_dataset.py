"""
Cross-dataset generalization test — วัดโมเดล Stage 2 (+pipeline) บนชุดข้อมูลเหล็กอื่น
ที่ "ไม่เคยเทรน" เพื่อประเมิน domain shift โดยไม่ต้องถ่ายภาพเอง

รองรับ annotation รูปแบบ COCO (Roboflow export):  <dir>/images/*.jpg + <dir>/*.coco.json
(หรือ <dir>/_annotations.coco.json)

วัด 3 อย่าง เทียบ pipeline (Stage1+2) vs baseline (Stage2 ภาพเต็ม):
  1. presence      — ภาพมีตำหนิ (GT>=1) ระบบตรวจเจอ >=1 กล่องไหม  (detection rate)
  2. loc-agnostic  — ต่อ GT box: มี pred box (คลาสใดก็ได้) IoU>=iou ไหม  (recall/precision เชิงตำแหน่ง)
  3. mapped        — เฉพาะคลาสที่ map ได้ (--map): P/R/F1 ระดับกล่อง ต้อง IoU>=iou และคลาสตรง

วิธีใช้:
    python evaluate_cross_dataset.py --dir external_test/gc10 --map external_test/gc10/classmap.json
    python evaluate_cross_dataset.py --dir external_test/gc10 --mode baseline --iou 0.3
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

import pipeline as P

BASE = Path(__file__).resolve().parent
CLASSES = P.DEFECT_CLASSES
NAME_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_coco(root: Path):
    cand = list(root.glob("*.coco.json")) + list(root.glob("_annotations.coco.json")) \
        + list(root.glob("*/_annotations.coco.json"))
    if not cand:
        raise SystemExit(f"ไม่พบไฟล์ COCO (.coco.json) ใน {root}")
    coco = json.loads(cand[0].read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    gt = defaultdict(list)   # file_name -> [(x1,y1,x2,y2, cat_name), ...]
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        gt[im["file_name"]].append((x, y, x + w, y + h, cats[a["category_id"]]))
    return gt, cats, cand[0].parent


def predict(image, s1, s2, conf, device, class_conf, use_stage1):
    """คืน list ของ (x1,y1,x2,y2, our_class_name) ในพิกัดภาพเต็ม + meta"""
    out = []
    if use_stage1:
        mask = P.run_stage1(s1, image, device)
        boxes, meta = P.build_regions(mask, image.shape)
    else:
        h, w = image.shape[:2]
        boxes, meta = [(0, 0, w, h)], {"metal_found": False, "fallback_full_image": True,
                                       "n_regions": 1, "metal_ratio": 0.0}
    region_dets = []
    for (x, y, w, h) in boxes:
        crop = image[y:y + h, x:x + w]
        for d in P.run_stage2(s2, crop, conf, device, class_conf=class_conf):
            cx1, cy1, cx2, cy2 = d["bbox_xyxy_crop"]
            d["bbox_xyxy_global"] = [cx1 + x, cy1 + y, cx2 + x, cy2 + y]
            region_dets.append(d)
    kept = P.cross_region_nms(region_dets, iou_thresh=0.5) if use_stage1 else region_dets
    for d in kept:
        out.append((*d["bbox_xyxy_global"], d["class"]))
    return out, meta


def evaluate(images, gt, s1, s2, conf, device, class_conf, use_stage1, iou_thr, cmap):
    n_img = len(images)
    presence_tp = presence_fn = 0
    la_matched = la_gt = la_pred = la_pred_hit = 0
    m_tp = defaultdict(int); m_fp = defaultdict(int); m_fn = defaultdict(int)
    meta_all = []
    t0 = time.perf_counter()
    for p in images:
        im = cv2.imread(str(p))
        if im is None:
            continue
        preds, meta = predict(im, s1, s2, conf, device, class_conf, use_stage1)
        meta_all.append(meta)
        gboxes = gt.get(p.name, [])

        # 1. presence
        if gboxes:
            if preds:
                presence_tp += 1
            else:
                presence_fn += 1

        # 2. loc-agnostic
        la_gt += len(gboxes)
        la_pred += len(preds)
        used = set()
        for gb in gboxes:
            hit = False
            for i, pb in enumerate(preds):
                if i in used:
                    continue
                if iou_xyxy(gb[:4], pb[:4]) >= iou_thr:
                    used.add(i); hit = True; break
            if hit:
                la_matched += 1
        for i, pb in enumerate(preds):
            if any(iou_xyxy(gb[:4], pb[:4]) >= iou_thr for gb in gboxes):
                la_pred_hit += 1

        # 3. mapped per-class
        if cmap:
            g_map = [(gb[0], gb[1], gb[2], gb[3], cmap.get(gb[4])) for gb in gboxes]
            g_map = [g for g in g_map if g[4]]
            p_used = set()
            for gb in g_map:
                matched = False
                for i, pb in enumerate(preds):
                    if i in p_used:
                        continue
                    if pb[4] == gb[4] and iou_xyxy(gb[:4], pb[:4]) >= iou_thr:
                        p_used.add(i); matched = True; break
                if matched:
                    m_tp[gb[4]] += 1
                else:
                    m_fn[gb[4]] += 1
            mapped_targets = set(cmap.values())
            for i, pb in enumerate(preds):
                if pb[4] in mapped_targets and i not in p_used:
                    m_fp[pb[4]] += 1
    dt = time.perf_counter() - t0

    la_recall = la_matched / la_gt if la_gt else 0.0
    la_prec = la_pred_hit / la_pred if la_pred else 0.0
    res = {
        "n_images": n_img,
        "sec_per_image": round(dt / max(n_img, 1), 3),
        "presence_recall": round(presence_tp / max(presence_tp + presence_fn, 1), 4),
        "loc_agnostic_recall": round(la_recall, 4),
        "loc_agnostic_precision": round(la_prec, 4),
        "n_gt_boxes": la_gt, "n_pred_boxes": la_pred,
        "stage1_metal_found_rate": round(sum(m["metal_found"] for m in meta_all) / max(len(meta_all), 1), 4),
        "fallback_rate": round(sum(m["fallback_full_image"] for m in meta_all) / max(len(meta_all), 1), 4),
    }
    if cmap:
        per = {}
        for cls in sorted(set(cmap.values())):
            tp, fp, fn = m_tp[cls], m_fp[cls], m_fn[cls]
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            per[cls] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4)}
        res["mapped_per_class"] = per
    return res


def show(title, r):
    print(f"\n===== {title} =====")
    print(f"  images {r['n_images']}  |  {r['sec_per_image']} s/img  |  "
          f"stage1_found {r['stage1_metal_found_rate']}  fallback {r['fallback_rate']}")
    print(f"  presence recall (เจอตำหนิอย่างน้อย 1 จุด/ภาพ) : {r['presence_recall']}")
    print(f"  loc-agnostic  recall {r['loc_agnostic_recall']}  precision {r['loc_agnostic_precision']}"
          f"   (GT {r['n_gt_boxes']} / pred {r['n_pred_boxes']} boxes)")
    if "mapped_per_class" in r:
        print(f"  {'mapped class':<16}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>8}{'R':>8}{'F1':>8}")
        for c, v in r["mapped_per_class"].items():
            print(f"  {c:<16}{v['tp']:>4}{v['fp']:>4}{v['fn']:>4}"
                  f"{v['precision']:>8}{v['recall']:>8}{v['f1']:>8}")


def main():
    ap = argparse.ArgumentParser(description="Cross-dataset generalization test (COCO input)")
    ap.add_argument("--dir", required=True, help="โฟลเดอร์ที่มี images/ + *.coco.json")
    ap.add_argument("--map", default=None,
                    help="JSON { 'their_class_name': 'our_class_name', ... } สำหรับ mapped per-class")
    ap.add_argument("--mode", default="both", choices=["pipeline", "baseline", "both"])
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-class-conf", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="จำกัดจำนวนภาพ (0=ทั้งหมด) — ไว้ทดสอบเร็ว")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.dir) if Path(args.dir).is_absolute() else BASE / args.dir
    gt, cats, coco_dir = load_coco(root)
    img_dir = root / "images" if (root / "images").is_dir() else coco_dir
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in P.IMAGE_EXTS and p.name in gt)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"ไม่พบภาพที่ match กับ annotation ใน {img_dir}")
    n_boxes = sum(len(gt.get(p.name, [])) for p in images)
    print(f"ชุด cross-dataset: {len(images)} ภาพ, {n_boxes} GT boxes")
    print(f"คลาสต้นทาง: {list(cats.values())}")

    cmap = None
    if args.map:
        cmap = json.loads(Path(args.map).read_text(encoding="utf-8"))
        bad = [v for v in cmap.values() if v not in NAME_TO_ID]
        if bad:
            raise SystemExit(f"--map มีคลาสปลายทางไม่รู้จัก: {bad} (ต้องเป็น {CLASSES})")
        print(f"class map: {cmap}")

    device = P.resolve_device(args.device)
    s1, s2 = P.load_models(device)
    class_conf = None if args.no_class_conf else P.load_class_conf()

    report = {"dir": str(root), "conf": args.conf, "iou": args.iou,
              "source_classes": list(cats.values()), "class_map": cmap}
    if args.mode in ("pipeline", "both"):
        report["pipeline"] = evaluate(images, gt, s1, s2, args.conf, device, class_conf,
                                      True, args.iou, cmap)
        show("PIPELINE (Stage 1 + Stage 2)", report["pipeline"])
    if args.mode in ("baseline", "both"):
        report["baseline"] = evaluate(images, gt, s1, s2, args.conf, device, class_conf,
                                      False, args.iou, cmap)
        show("BASELINE (Stage 2 ภาพเต็ม)", report["baseline"])
    if args.mode == "both":
        a, b = report["pipeline"], report["baseline"]
        print("\n===== เทียบ =====")
        for k in ("presence_recall", "loc_agnostic_recall", "loc_agnostic_precision"):
            print(f"  {k:<24}{a[k]:>10}{b[k]:>10}{a[k] - b[k]:>+10.4f}")

    out = args.out or (BASE / "results" / f"crossdataset_{root.name}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nบันทึก: {out}")


if __name__ == "__main__":
    main()
