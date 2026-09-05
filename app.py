"""
Prototype UI: อัปโหลดภาพเหล็ก -> ระบบตรวจ 2 ขั้นตอน -> แสดงกรอบเหล็ก + ตารางตำหนิ

รัน:
    python app.py
เปิดเบราว์เซอร์ที่ http://127.0.0.1:7860

ต้องมี gradio:  pip install gradio
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import pipeline as P

try:
    import gradio as gr
except ImportError:
    sys.exit("ยังไม่ได้ติดตั้ง gradio — รัน: pip install gradio")

BASE_DIR = Path(__file__).resolve().parent

_STATE = {"s1": None, "s2": None, "device": None, "class_conf": None}

# ระดับความเสี่ยง -> ลำดับการแสดงผล (สูงก่อน) + สี badge
_RISK_ORDER = {"สูง": 0, "ปานกลาง-สูง": 1, "ปานกลาง": 2, "ต่ำ-ปานกลาง": 3, "ต่ำ": 4}


def _ensure_models():
    if _STATE["s1"] is None:
        _STATE["device"] = P.resolve_device("auto")
        _STATE["s1"], _STATE["s2"] = P.load_models(_STATE["device"])
        _STATE["class_conf"] = P.load_class_conf()
        if _STATE["class_conf"]:
            print("ใช้ per-class conf จาก thresholds.json:", _STATE["class_conf"])
    return _STATE["s1"], _STATE["s2"], _STATE["device"]


def _to_bgr(image_rgb):
    """รับ array จาก gradio (อาจเป็น grayscale / RGBA / RGB) -> BGR 3 ช่อง"""
    arr = np.asarray(image_rgb)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _stage1_view(image_bgr, mask, boxes, meta):
    """ภาพแสดงผล Stage 1: เน้นพื้นที่ที่เป็นเหล็ก (เขียว) + กรอบ region"""
    view = image_bgr.copy()
    green = np.zeros_like(view)
    green[:, :] = (0, 200, 0)
    m = mask.astype(bool)
    view[m] = cv2.addWeighted(view, 0.55, green, 0.45, 0)[m]
    n_real = meta["n_regions"] - (1 if meta["fallback_full_image"] else 0)
    for i, (x, y, w, h) in enumerate(boxes):
        is_full = meta["fallback_full_image"] and i == len(boxes) - 1
        col = (0, 165, 255) if is_full else (0, 255, 0)  # ส้ม = fallback ทั้งภาพ
        cv2.rectangle(view, (x, y), (x + w, y + h), col, 2)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def analyze(image_rgb, conf, detailed, progress=gr.Progress()):
    if image_rgb is None:
        return None, None, "### อัปโหลดภาพก่อน", []

    progress(0.1, desc="โหลดโมเดล...")
    s1, s2, device = _ensure_models()
    image_bgr = _to_bgr(image_rgb)

    # ----- Stage 1 : หาพื้นที่ที่เป็นเหล็ก + fallback (เหมือน pipeline.py) -----
    progress(0.35, desc="Stage 1: หาพื้นที่เหล็ก...")
    mask = P.run_stage1(s1, image_bgr, device)
    boxes, meta = P.build_regions(mask, image_bgr.shape)
    metal_ratio = meta["metal_ratio"]
    n_metal = meta["n_regions"] - (1 if meta["fallback_full_image"] else 0)
    stage1_img = _stage1_view(image_bgr, mask, boxes, meta)

    # ----- Stage 2 : ตรวจตำหนิทุกบริเวณ + แปลง bbox เป็นพิกัดภาพเต็ม -----
    progress(0.55, desc="Stage 2: ตรวจตำหนิ...")
    region_dets = []
    for x, y, w, h in boxes:
        crop = image_bgr[y:y + h, x:x + w]
        dets = P.run_stage2(s2, crop, conf, device, augment=bool(detailed),
                            class_conf=_STATE["class_conf"])
        for d in dets:
            cx1, cy1, cx2, cy2 = d["bbox_xyxy_crop"]
            d["bbox_xyxy_global"] = [cx1 + x, cy1 + y, cx2 + x, cy2 + y]
        region_dets.append(dets)

    # ----- Cross-region NMS: ตัด detection ซ้ำจากกรอบทับกัน / fallback ทั้งภาพ -----
    progress(0.8, desc="รวมผล...")
    flat = [d for dets in region_dets for d in dets]
    kept_ids = {id(d) for d in P.cross_region_nms(flat, iou_thresh=0.5)}

    annotated = image_bgr.copy()
    rows = []
    for i, (x, y, w, h) in enumerate(boxes):
        detections = [d for d in region_dets[i] if id(d) in kept_ids]
        is_full = meta["fallback_full_image"] and i == len(boxes) - 1
        tag = "ทั้งภาพ" if is_full else f"#{i + 1}"
        box_col = (0, 165, 255) if is_full else (0, 255, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_col, 2)

        if detections:
            top = detections[0]
            info = P.DEFECT_INFO[top["class"]]
            annotated = P.draw_thai_text(
                annotated, f"{info['name_th']} ({top['confidence']:.0%})",
                (x, y - 28), color_bgr=(0, 0, 255),
            )
            for d in detections:
                gx1, gy1, gx2, gy2 = (int(v) for v in d["bbox_xyxy_global"])
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                di = P.DEFECT_INFO[d["class"]]
                rows.append([tag, d["class"], di["name_th"],
                             f"{d['confidence']:.1%}", di["risk"]])
        else:
            annotated = P.draw_thai_text(annotated, f"เหล็ก {tag} ปกติ",
                                         (x, y - 28), color_bgr=(0, 150, 0))

    rows.sort(key=lambda r: (_RISK_ORDER.get(r[4], 9), -float(r[3].rstrip("%"))))

    # ----- สรุปผล (headline) -----
    if not rows:
        verdict = "## ✅ ไม่พบตำหนิ\nระบบไม่พบตำหนิพื้นผิวในภาพนี้ (ที่ค่าความมั่นใจ ≥ %.2f)" % conf
    else:
        risky = sorted({r[2] for r in rows if r[4] in ("สูง", "ปานกลาง-สูง")},
                       key=lambda n: n)
        kinds = sorted({r[2] for r in rows})
        head = "## ⚠️ พบตำหนิ %d ชนิด: %s" % (len(kinds), ", ".join(kinds))
        if risky:
            head += "\n### 🔴 ความเสี่ยงสูง: %s" % ", ".join(risky)
        verdict = head

    fb = ("  \n> Stage 1 เจอเหล็กน้อย (%.0f%%) — เพิ่มการตรวจทั้งภาพเป็น fallback"
          % (metal_ratio * 100)) if meta["fallback_full_image"] else ""
    if _STATE["class_conf"]:
        fb += "  \n> ใช้ threshold รายคลาส (thresholds.json) — สไลเดอร์เป็นค่าขั้นต่ำ"
    verdict += ("\n\n`Stage 1: %d บริเวณ / เหล็กครอบคลุม %.0f%%`  "
                "`Stage 2: %d จุด`  `device: %s`%s"
                % (n_metal, metal_ratio * 100, len(rows), device, fb))

    return (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), stage1_img, verdict, rows)


def build_ui():
    sample_dir = BASE_DIR / "test_images"
    samples = [[str(p)] for p in sorted(sample_dir.glob("*")) if p.suffix.lower() in P.IMAGE_EXTS] \
        if sample_dir.exists() else []

    with gr.Blocks(title="Steel Defect Detection (2-stage)") as demo:
        gr.Markdown(
            "# ตรวจจับตำหนิพื้นผิวเหล็ก — Prototype\n"
            "**Stage 1** DMS46 หาพื้นที่ที่เป็นเหล็ก → **Stage 2** YOLO11s (train-gray-s) ตรวจชนิดตำหนิ 8 ประเภท "
            "— crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches, rust, crack\n"
            "> prototype ผู้ช่วยคัดกรอง — ไม่ใช่ระบบตรวจสอบใช้งานจริง"
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="ภาพเหล็กที่จะตรวจ", height=360)
                conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                 label="Confidence threshold (Stage 2)")
                detailed = gr.Checkbox(
                    value=False,
                    label="ตรวจละเอียด (test-time augmentation — ช้าลง ~2-3x, recall ดีขึ้นเล็กน้อย)")
                btn = gr.Button("ตรวจสอบ", variant="primary")
                if samples:
                    gr.Examples(examples=samples, inputs=inp, label="ตัวอย่างภาพ")
            with gr.Column(scale=1):
                verdict = gr.Markdown("อัปโหลดภาพแล้วกด **ตรวจสอบ**")
                out_tbl = gr.Dataframe(
                    headers=["บริเวณ", "class", "ชนิด (ไทย)", "ความมั่นใจ", "ความเสี่ยง"],
                    label="รายการตำหนิที่พบ (เรียงตามความเสี่ยง)",
                    interactive=False, wrap=True,
                )
        with gr.Row():
            out_img = gr.Image(type="numpy", label="ผลลัพธ์ (กรอบเขียว = บริเวณเหล็ก, กรอบแดง = ตำหนิ)")
            out_s1 = gr.Image(type="numpy", label="Stage 1 — พื้นที่ที่เป็นเหล็ก (เขียว) / fallback (ส้ม)")

        btn.click(analyze, inputs=[inp, conf, detailed],
                  outputs=[out_img, out_s1, verdict, out_tbl])
    return demo


def _lan_ip():
    """เดา IP วง LAN ของเครื่องนี้ (ไว้บอกคนอื่นในวง wifi/สาย เดียวกันให้เปิดตาม)"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # ไม่ได้ส่งข้อมูลจริง แค่ให้ OS เลือก interface ที่ออกเน็ตได้
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--share", action="store_true",
                     help="สร้างลิงก์สาธารณะ *.gradio.live ผ่านทันเนลของ Gradio "
                          "(ไว้ให้คนไม่ได้อยู่วง wifi/LAN เดียวกันทดลองใช้ได้ — ลิงก์อยู่ได้ 72 ชม. "
                          "ภาพที่อัปโหลดจะผ่านเซิร์ฟเวอร์ของ Gradio ด้วย)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--local-only", action="store_true",
                     help="เปิดเฉพาะเครื่องนี้ (127.0.0.1) — ค่าเริ่มต้นคือเปิดทั้งวง LAN (0.0.0.0) "
                          "ให้คนที่ต่อ wifi/สายเดียวกันเข้าได้ด้วย")
    args = ap.parse_args()

    print("กำลังเตรียมโมเดล (โหลดครั้งเดียวตอนเริ่ม)...")
    _ensure_models()

    host = "127.0.0.1" if args.local_only else "0.0.0.0"
    print(f"พร้อมใช้งาน — เปิดเองที่ http://127.0.0.1:{args.port}")
    if not args.local_only:
        lan_ip = _lan_ip()
        if lan_ip:
            print(f"คนอื่นในวง wifi/LAN เดียวกัน เปิดที่ http://{lan_ip}:{args.port}")
        print("  (ถ้าเข้าจากเครื่องอื่นไม่ได้ ให้เช็ค Windows Firewall — "
              f"ต้องอนุญาต inbound พอร์ต {args.port} สำหรับเครือข่ายส่วนตัว/Private)")
    if args.share:
        print("กำลังสร้างลิงก์สาธารณะ *.gradio.live ... (รอสักครู่)")
    print()

    build_ui().queue().launch(server_name=host, server_port=args.port, share=args.share)
