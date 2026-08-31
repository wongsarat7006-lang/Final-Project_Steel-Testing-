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

_STATE = {"s1": None, "s2": None, "device": None}


def _ensure_models():
    if _STATE["s1"] is None:
        _STATE["device"] = P.resolve_device("auto")
        _STATE["s1"], _STATE["s2"] = P.load_models(_STATE["device"])
    return _STATE["s1"], _STATE["s2"], _STATE["device"]


def analyze(image_rgb, conf):
    if image_rgb is None:
        return None, [], "อัปโหลดภาพก่อน"

    s1, s2, device = _ensure_models()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    mask = P.run_stage1(s1, image_bgr, device)
    boxes = P.mask_to_boxes(mask)
    metal_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])

    annotated = image_bgr.copy()
    table = []

    if not boxes:
        msg = f"Stage 1: ไม่พบพื้นที่ที่เป็นเหล็กในภาพ (เหล็กครอบคลุม {metal_ratio:.0%})"
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), table, msg

    total_defects = 0
    for i, (x, y, w, h) in enumerate(boxes):
        crop = image_bgr[y:y + h, x:x + w]
        detections = P.run_stage2(s2, crop, conf, device)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if detections:
            top = detections[0]
            info = P.DEFECT_INFO[top["class"]]
            annotated = P.draw_thai_text(
                annotated, f"{info['name_th']} ({top['confidence']:.0%})",
                (x, y - 28), color_bgr=(0, 0, 255),
            )
            for d in detections:
                di = P.DEFECT_INFO[d["class"]]
                table.append([f"#{i + 1}", d["class"], di["name_th"],
                              f"{d['confidence']:.1%}", di["risk"]])
                total_defects += 1
        else:
            annotated = P.draw_thai_text(annotated, f"เหล็ก #{i + 1} ปกติ",
                                         (x, y - 28), color_bgr=(0, 150, 0))
            table.append([f"#{i + 1}", "-", "ไม่พบตำหนิ", "-", "-"])

    msg = (f"Stage 1: พบเหล็ก {len(boxes)} บริเวณ (ครอบคลุม {metal_ratio:.0%})  |  "
           f"Stage 2: พบตำหนิรวม {total_defects} จุด  |  device: {device}")
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), table, msg


def build_ui():
    sample_dir = BASE_DIR / "test_images"
    samples = [[str(p)] for p in sorted(sample_dir.glob("*")) if p.suffix.lower() in P.IMAGE_EXTS] \
        if sample_dir.exists() else []

    with gr.Blocks(title="Steel Defect Detection (2-stage)") as demo:
        gr.Markdown(
            "# ตรวจจับตำหนิพื้นผิวเหล็ก — Prototype\n"
            "**Stage 1** DMS46 หาพื้นที่ที่เป็นเหล็ก → **Stage 2** YOLO11 ตรวจชนิดตำหนิ 8 ประเภท "
            "(crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches, rust, crack)"
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="ภาพเหล็กที่จะตรวจ")
                conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Confidence threshold (Stage 2)")
                btn = gr.Button("ตรวจสอบ", variant="primary")
                if samples:
                    gr.Examples(examples=samples, inputs=inp, label="ตัวอย่างภาพ")
            with gr.Column():
                out_img = gr.Image(type="numpy", label="ผลลัพธ์")
                out_msg = gr.Textbox(label="สรุป", interactive=False)
                out_tbl = gr.Dataframe(
                    headers=["บริเวณ", "class", "ชนิด (ไทย)", "ความมั่นใจ", "ความเสี่ยง"],
                    label="รายการตำหนิที่พบ", interactive=False, wrap=True,
                )
        btn.click(analyze, inputs=[inp, conf], outputs=[out_img, out_tbl, out_msg])
    return demo


if __name__ == "__main__":
    build_ui().launch()
