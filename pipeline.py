"""
Pipeline 2 ขั้นตอน: หาตำแหน่งเหล็ก (Stage 1) + ตรวจตำหนิ (Stage 2)

  Stage 1  ->  DMS46 (Apple Dense Material Segmentation) หา region ที่เป็นวัสดุ "Metal"
  Stage 2  ->  YOLO11 (เทรนเอง) ตรวจชนิดตำหนิบนภาพเหล็กที่ crop มา

วิธีใช้:
    python pipeline.py --image test_images/steel-plate3.jpg
    python pipeline.py --folder test_images --output_dir pipeline_results
    python pipeline.py --image a.jpg --device cpu --conf 0.35

ผลลัพธ์: ภาพ *_result.jpg (กรอบเหล็ก + ป้ายตำหนิภาษาไทย) และไฟล์สรุป *_result.json
"""
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ===== Path (อ้างอิงจากตำแหน่งไฟล์นี้ ไม่ผูกกับ working directory) =====
BASE_DIR = Path(__file__).resolve().parent
STAGE1_MODEL_PATH = BASE_DIR / "DMS46_v1.pt"
STAGE2_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train-gray-s" / "weights" / "best.pt"
THRESHOLDS_PATH = BASE_DIR / "thresholds.json"  # per-class conf (ถ้ามี) — สร้างด้วย tune_thresholds.py

# DMS46 ทำนายเป็น index 0-45 (เรียงจาก taxonomy 46 ชนิดที่โมเดลรองรับ)
# วัสดุ "Metal" คือ taxonomy id 26 ซึ่งตรงกับ output index 22 ของโมเดล
# (ดู dms46 list ใน ml-dms-dataset/inference.py: ตำแหน่งของเลข 26 คือ index 22)
METAL_MODEL_INDEX = 22

# ImageNet normalization แบบเดียวกับ ml-dms-dataset/inference.py (input เป็น 0-255 float)
_IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
_IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]
_DMS_INPUT_DIM = 512

# ต้องเรียงตามลำดับคลาสของโมเดล Stage 2 (merged_dataset/data.yaml)
DEFECT_CLASSES = [
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled-in_scale", "scratches", "rust", "crack",
]

DEFECT_INFO = {
    "crazing": {"name_th": "รอยแตกลายงา", "risk": "ปานกลาง-สูง"},
    "inclusion": {"name_th": "สิ่งแปลกปลอมฝังใน", "risk": "ปานกลาง"},
    "patches": {"name_th": "รอยแผ่น/ผิวลอก", "risk": "ต่ำ-ปานกลาง"},
    "pitted_surface": {"name_th": "ผิวขรุขระเป็นหลุม", "risk": "ปานกลาง"},
    "rolled-in_scale": {"name_th": "สะเก็ดฝังจากการรีด", "risk": "ปานกลาง"},
    "scratches": {"name_th": "รอยขีดข่วน", "risk": "ต่ำ"},
    "rust": {"name_th": "สนิม", "risk": "สูง"},
    "crack": {"name_th": "รอยแตกร้าว", "risk": "สูง"},
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_device(requested: str) -> str:
    """เลือก device: 'auto' -> cuda ถ้ามี ไม่งั้น cpu; ถ้าขอ cuda แต่ไม่มีให้เตือนแล้ว fallback"""
    if requested == "cpu":
        return "cpu"
    if requested in ("cuda", "0", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if requested != "auto":
            print("ไม่พบ CUDA GPU — เปลี่ยนไปใช้ CPU แทน")
        return "cpu"
    return requested


def draw_thai_text(img_bgr, text, position, color_bgr=(0, 0, 255), font_size=22):
    """วาดข้อความไทยลงภาพ (cv2.putText ไม่รองรับไทย จึงวาดผ่าน PIL)"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\tahoma.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    x, y = position
    y = max(y, 0)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    text_bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad],
        fill=(255, 255, 255),
    )
    draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def load_models(device: str):
    print(f"กำลังโหลดโมเดล (device={device})...")
    if not STAGE1_MODEL_PATH.exists():
        raise FileNotFoundError(f"ไม่พบโมเดล Stage 1: {STAGE1_MODEL_PATH}")
    if not STAGE2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบโมเดล Stage 2: {STAGE2_MODEL_PATH}\n"
            f"เทรนก่อนด้วย: python train.py"
        )

    print("  Stage 1 (Metal Localization / DMS46)...")
    stage1_model = torch.jit.load(str(STAGE1_MODEL_PATH), map_location=device)
    stage1_model.eval()
    if device == "cuda":
        stage1_model = stage1_model.cuda()

    print("  Stage 2 (Defect Detection / YOLO11)...")
    stage2_model = YOLO(str(STAGE2_MODEL_PATH))

    # ตรวจว่าโมเดล Stage 2 เทรนบน grayscale หรือไม่ (จาก data yaml ที่ใช้เทรน)
    # ถ้าใช่ ต้องแปลง crop เป็น grayscale ก่อนตรวจ ไม่งั้น train/serve skew
    try:
        train_data = str((getattr(stage2_model, "ckpt", None) or {})
                         .get("train_args", {}).get("data", ""))
    except Exception:
        train_data = ""
    stage2_model._steel_gray = "gray" in train_data.lower()
    if stage2_model._steel_gray:
        print("  (Stage 2 เทรนบน grayscale → จะแปลง crop เป็น grayscale ก่อนตรวจ)")

    print("โหลดโมเดลครบแล้ว\n")
    return stage1_model, stage2_model


def _preprocess_for_dms(image_bgr):
    """resize รักษาสัดส่วน (ด้านยาวสุด = 512) + ImageNet normalize เหมือน inference.py ต้นฉบับ"""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(_DMS_INPUT_DIM / h, _DMS_INPUT_DIM / w)
    new_h, new_w = math.ceil(scale * h), math.ceil(scale * w)

    resized = np.array(Image.fromarray(img_rgb).resize((new_w, new_h), Image.LANCZOS))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()  # 0-255, ไม่หาร 255
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def run_stage1(stage1_model, image_bgr, device: str):
    """คืน mask (0/255) ขนาดเท่าภาพต้นฉบับ: 255 = พื้นที่ที่เป็นเหล็ก"""
    tensor = _preprocess_for_dms(image_bgr)
    if device == "cuda":
        tensor = tensor.cuda()

    with torch.no_grad():
        output = stage1_model(tensor)

    # DMS46 คืน tuple/list ความยาว 1, output[0] shape [1,1,H,W] เป็น label map (argmax แล้ว)
    if isinstance(output, (tuple, list)):
        output = output[0]
    pred = output.squeeze().cpu().numpy().astype(np.int64)  # [H, W]

    metal_mask = (pred == METAL_MODEL_INDEX).astype(np.uint8) * 255
    metal_mask = cv2.resize(
        metal_mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return metal_mask


def _merge_close_boxes(boxes, gap):
    """รวมกรอบที่ทับกันหรืออยู่ห่างกันไม่เกิน gap พิกเซล (วนจนไม่มีอะไรให้รวม)"""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged = []
        while boxes:
            x, y, w, h = boxes.pop()
            ax1, ay1, ax2, ay2 = x, y, x + w, y + h
            keep = []
            for bx, by, bw, bh in boxes:
                bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
                if (ax1 - gap < bx2 and bx1 - gap < ax2 and
                        ay1 - gap < by2 and by1 - gap < ay2):
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    changed = True
                else:
                    keep.append((bx, by, bw, bh))
            boxes = keep
            merged.append([ax1, ay1, ax2 - ax1, ay2 - ay1])
        boxes = merged
    return [tuple(int(v) for v in b) for b in boxes]


def mask_to_boxes(mask, min_area=500, max_area_ratio=0.98, min_fill_ratio=0.10):
    """แปลง mask เป็น bounding box: รวมกรอบที่อยู่ติดกัน + ตัดกรอบจิ๋วทิ้งถ้ามีกรอบใหญ่แล้ว"""
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    img_h, img_w = mask.shape[:2]
    img_area = img_h * img_w

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = w * h
        if box_area == 0:
            continue
        if box_area / img_area > max_area_ratio and area / box_area < 0.5:
            continue
        if area / box_area < min_fill_ratio:
            continue
        raw.append((int(x), int(y), int(w), int(h)))

    if not raw:
        if cv2.countNonZero(clean) > min_area:
            ys, xs = np.where(clean > 0)
            return [(int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))]
        return []

    # รวมกรอบที่อยู่ห่างกันไม่เกิน ~3% ของด้านสั้น (เศษกรอบของวัตถุชิ้นเดียวกัน)
    gap = max(8, int(0.03 * min(img_h, img_w)))
    boxes = _merge_close_boxes(raw, gap)

    # ถ้ามีกรอบใหญ่ (>5% ของภาพ) อยู่แล้ว ให้ทิ้งกรอบจิ๋ว (<1%)
    if any(w * h > 0.05 * img_area for (_, _, w, h) in boxes):
        boxes = [b for b in boxes if b[2] * b[3] >= 0.01 * img_area]
    return boxes


def build_regions(mask, image_shape, min_metal_ratio=0.05):
    """หา region เหล็กจาก mask + เพิ่มกรอบทั้งภาพเป็น fallback เมื่อ Stage 1
    เจอเหล็กน้อยกว่า min_metal_ratio หรือไม่เจอเลย (DMS46 ทำงานไม่ดีกับ
    เหล็กทาสี/สนิมหนัก/ภาพ close-up texture — ดู evaluate_stage1.py)

    คืน (boxes, meta):
      boxes : list[(x, y, w, h)] พิกัดในระบบภาพเต็ม (มีกรอบทั้งภาพต่อท้ายถ้า fallback)
      meta  : dict สรุปว่า Stage 1 ทำงานอย่างไรกับภาพนี้
    ใช้ร่วมกันโดย pipeline.py / evaluate.py / evaluate_real.py / app.py
    เพื่อให้ fallback เหมือนกันทุกที่"""
    boxes = mask_to_boxes(mask)
    img_h, img_w = image_shape[:2]
    metal_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
    metal_found = len(boxes) > 0
    fallback = (not boxes) or metal_ratio < min_metal_ratio
    if fallback:
        boxes = boxes + [(0, 0, img_w, img_h)]
    return boxes, {
        "metal_found": metal_found,
        "metal_ratio": round(metal_ratio, 4),
        "fallback_full_image": fallback,
        "n_regions": len(boxes),
    }


def load_class_conf(path=None):
    """โหลด per-class confidence threshold จาก thresholds.json (สร้างด้วย tune_thresholds.py)
    คืน dict {class_name: conf} หรือ None ถ้าไม่มีไฟล์"""
    path = Path(path) if path else THRESHOLDS_PATH
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v["conf"]) for k, v in obj.get("per_class", {}).items()}


def run_stage2(stage2_model, crop_image, conf: float, device: str, augment: bool = False,
               class_conf: dict | None = None):
    """รัน YOLO ตรวจตำหนิบน crop, คืน list ของ detection เรียงตาม confidence มาก->น้อย
    (bbox เป็นพิกัดของ crop — ผู้เรียกต้องบวก offset ของ region เองถ้าจะเทียบข้ามบริเวณ)
    augment=True    : test-time augmentation ของ ultralytics (ช้าลง ~2-3x, recall ดีขึ้นเล็กน้อย)
    class_conf={..} : กรอง detection ด้วย threshold รายคลาส (predict ที่ค่าต่ำสุดก่อน แล้วค่อยกรอง)"""
    if crop_image.size == 0 or crop_image.shape[0] < 8 or crop_image.shape[1] < 8:
        return []
    if getattr(stage2_model, "_steel_gray", False):
        g = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        crop_image = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    base_conf = min([conf, *class_conf.values()]) if class_conf else conf
    results = stage2_model.predict(
        source=crop_image, conf=base_conf, device=device, augment=augment, verbose=False
    )
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = DEFECT_CLASSES[cls_id]
            score = float(box.conf[0])
            if class_conf and score < class_conf.get(name, conf):
                continue
            detections.append({
                "class": name,
                "confidence": score,
                "bbox_xywh": [round(v, 1) for v in box.xywh[0].tolist()],
                "bbox_xyxy_crop": [round(v, 1) for v in box.xyxy[0].tolist()],
            })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def _iou_xyxy(a, b):
    """IoU ของกรอบสองอันในรูปแบบ [x1, y1, x2, y2]"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def cross_region_nms(detections, iou_thresh=0.5):
    """class-aware NMS ข้าม region: ตัด detection ที่ซ้ำกันเพราะกรอบ metal ทับกัน
    หรือเพราะ fallback ตรวจทั้งภาพซ้อนกับกรอบ metal.
    detection แต่ละตัวต้องมี key 'bbox_xyxy_global' (พิกัดในระบบภาพเต็ม) และ 'confidence'.
    คืน list ที่รอด เรียงตาม confidence มาก->น้อย"""
    order = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in order:
        if any(d["class"] == k["class"]
               and _iou_xyxy(d["bbox_xyxy_global"], k["bbox_xyxy_global"]) > iou_thresh
               for k in kept):
            continue
        kept.append(d)
    return kept


def process_image(image_path, stage1_model, stage2_model, output_dir, conf, device,
                  min_metal_ratio=0.05, nms_iou=0.5, class_conf=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(image_path).stem

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"เปิดภาพไม่ได้: {image_path}")
        return None

    print(f"===== {filename} =====")
    print("Stage 1: หาตำแหน่งเหล็ก...")
    mask = run_stage1(stage1_model, image, device)
    boxes, meta = build_regions(mask, image.shape, min_metal_ratio)
    n_metal = meta["n_regions"] - (1 if meta["fallback_full_image"] else 0)
    print(f"  พบเหล็ก {n_metal} บริเวณ (ครอบคลุม {meta['metal_ratio']:.0%} ของภาพ)")
    if meta["fallback_full_image"]:
        print(f"  เหล็กครอบคลุมน้อยกว่า {min_metal_ratio:.0%} — เพิ่มการตรวจทั้งภาพ (fallback)")

    summary = {
        "image": str(image_path),
        "metal_regions": len(boxes),
        "metal_area_ratio": meta["metal_ratio"],
        "fallback_full_image": meta["fallback_full_image"],
        "regions": [],
    }

    # ----- Pass 1: ตรวจตำหนิทุกบริเวณ แปลง bbox เป็นพิกัดภาพเต็ม -----
    region_dets = []
    for i, (x, y, w, h) in enumerate(boxes):
        crop = image[y:y + h, x:x + w]
        print(f"Stage 2: ตรวจตำหนิบริเวณ {i + 1}/{len(boxes)}...")
        dets = run_stage2(stage2_model, crop, conf, device, class_conf=class_conf)
        for d in dets:
            cx1, cy1, cx2, cy2 = d["bbox_xyxy_crop"]
            d["region_id"] = i
            d["bbox_xyxy_global"] = [
                round(cx1 + x, 1), round(cy1 + y, 1),
                round(cx2 + x, 1), round(cy2 + y, 1),
            ]
        region_dets.append(dets)

    # ----- Cross-region NMS: ตัด detection ซ้ำจากกรอบที่ทับกัน / fallback ทั้งภาพ -----
    flat = [d for dets in region_dets for d in dets]
    kept_ids = {id(d) for d in cross_region_nms(flat, iou_thresh=nms_iou)}
    n_removed = len(flat) - len(kept_ids)
    if n_removed:
        print(f"  ตัด detection ซ้ำข้ามบริเวณ {n_removed} จุด (cross-region NMS)")

    # ----- Pass 2: วาดผล + สรุป เฉพาะ detection ที่รอด NMS -----
    annotated = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        detections = [d for d in region_dets[i] if id(d) in kept_ids]

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if detections:
            top = detections[0]
            info = DEFECT_INFO[top["class"]]
            label = f"{info['name_th']} ({top['confidence']:.0%})"
            annotated = draw_thai_text(annotated, label, (x, y - 28), color_bgr=(0, 0, 255))
            for d in detections:
                di = DEFECT_INFO[d["class"]]
                print(f"    - {di['name_th']} ({d['class']}) conf {d['confidence']:.0%} "
                      f"ความเสี่ยง {di['risk']}")
        else:
            annotated = draw_thai_text(annotated, f"เหล็ก #{i + 1} ปกติ", (x, y - 28),
                                       color_bgr=(0, 150, 0))
            print("    ไม่พบตำหนิ (สภาพดี)")

        summary["regions"].append({
            "region_id": i,
            "box_xywh": [x, y, w, h],
            "detections": detections,
        })

    out_img = output_dir / f"{filename}_result.jpg"
    out_json = output_dir / f"{filename}_result.json"
    cv2.imwrite(str(out_img), annotated)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"บันทึก: {out_img.name}, {out_json.name}\n")
    return summary


def iter_images(folder):
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def main():
    parser = argparse.ArgumentParser(description="Steel defect detection pipeline (2-stage)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path ของภาพเดียว")
    src.add_argument("--folder", help="path ของโฟลเดอร์ (ประมวลผลทุกภาพ)")
    parser.add_argument("--output_dir", default="pipeline_results")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold ของ Stage 2")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--min-metal-ratio", type=float, default=0.05,
                        help="ถ้า Stage 1 เจอเหล็กน้อยกว่านี้ ให้ตรวจทั้งภาพเพิ่ม (fallback)")
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="IoU threshold ของ cross-region NMS (ตัด detection ซ้ำข้ามบริเวณ)")
    parser.add_argument("--no-class-conf", action="store_true",
                        help="ไม่ใช้ per-class threshold จาก thresholds.json (ใช้ --conf ค่าเดียว)")
    parser.add_argument("--weights", default=None,
                        help="path ของ Stage 2 .pt (default: STAGE2_MODEL_PATH ในไฟล์นี้)")
    args = parser.parse_args()

    if args.weights:
        global STAGE2_MODEL_PATH
        w = Path(args.weights)
        if not w.is_file():
            raise SystemExit(f"ไม่พบไฟล์ weights: {w}")
        STAGE2_MODEL_PATH = w
        print(f"ใช้ Stage 2 weights: {w}")

    device = resolve_device(args.device)
    s1, s2 = load_models(device)

    class_conf = None if args.no_class_conf else load_class_conf()
    if class_conf:
        print(f"ใช้ per-class conf จาก {THRESHOLDS_PATH.name}: "
              + ", ".join(f"{k}={v}" for k, v in class_conf.items()) + "\n")

    if args.image:
        process_image(args.image, s1, s2, args.output_dir, args.conf, device,
                      args.min_metal_ratio, args.nms_iou, class_conf)
    else:
        images = list(iter_images(args.folder))
        if not images:
            print(f"ไม่พบไฟล์ภาพใน {args.folder}")
            return
        print(f"พบ {len(images)} ภาพใน {args.folder}\n")
        all_summaries = []
        for img in images:
            s = process_image(img, s1, s2, args.output_dir, args.conf, device,
                              args.min_metal_ratio, args.nms_iou, class_conf)
            if s:
                all_summaries.append(s)
        index_path = Path(args.output_dir) / "_index.json"
        index_path.write_text(
            json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_defect = sum(1 for s in all_summaries
                       for r in s["regions"] if r["detections"])
        print(f"เสร็จ: {len(all_summaries)} ภาพ, พบบริเวณที่มีตำหนิรวม {n_defect} จุด")
        print(f"สรุปทั้งหมด: {index_path}")


if __name__ == "__main__":
    main()
