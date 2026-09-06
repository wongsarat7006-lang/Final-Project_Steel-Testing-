"""
Prototype UI: อัปโหลดภาพเหล็ก -> ระบบตรวจ 2 ขั้นตอน -> แสดงกรอบเหล็ก + ตารางตำหนิ

รัน:
    python app.py                 # เปิดที่ http://127.0.0.1:7860 (และวง LAN เดียวกัน)
    python app.py --share         # เพิ่มลิงก์สาธารณะ *.gradio.live
    python app.py --local-only    # เปิดเฉพาะเครื่องนี้

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

# ระดับความเสี่ยง -> ลำดับการแสดงผล (สูงก่อน) + สีชิป
_RISK_ORDER = {"สูง": 0, "ปานกลาง-สูง": 1, "ปานกลาง": 2, "ต่ำ-ปานกลาง": 3, "ต่ำ": 4}
_RISK_COLOR = {
    "สูง": ("#fdecea", "#b71c1c"),
    "ปานกลาง-สูง": ("#fff0e6", "#c05600"),
    "ปานกลาง": ("#fff8e1", "#8a6d00"),
    "ต่ำ-ปานกลาง": ("#e8f0fe", "#1a56c4"),
    "ต่ำ": ("#eef1f4", "#5a6270"),
}
_HIGH_RISK = ("สูง", "ปานกลาง-สูง")

# โหมดความไว -> ตัวคูณ threshold รายคลาส (ยิ่งต่ำ = ยิ่งเตือนเยอะ)
_SENS = {"มาตรฐาน": 1.0, "ไว": 0.6, "ไวมาก": 0.4}
_RAW_FLOOR = 0.12       # conf ต่ำสุดที่ให้ YOLO คืนมา
_TENTATIVE_FLOOR = 0.30  # ต่ำกว่า threshold แต่ >= ค่านี้ = "อาจมี" (แสดงแยก)


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
    for i, (x, y, w, h) in enumerate(boxes):
        is_full = meta["fallback_full_image"] and i == len(boxes) - 1
        col = (0, 165, 255) if is_full else (0, 255, 0)  # ส้ม = fallback ทั้งภาพ
        cv2.rectangle(view, (x, y), (x + w, y + h), col, 2)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


# ---------- HTML rendering ----------
def _banner(bg, fg, text, sub=""):
    sub_html = f"<div style='font-size:13px;font-weight:400;margin-top:4px;opacity:.85'>{sub}</div>" if sub else ""
    return (f"<div style='padding:16px 20px;border-radius:12px;background:{bg};color:{fg};"
            f"font-size:19px;font-weight:700;line-height:1.35'>{text}{sub_html}</div>")


def _empty_banner():
    return _banner("#eef1f4", "#5a6270", "⬆️ อัปโหลดภาพเหล็ก แล้วกด “ตรวจสอบ”",
                   "หรือเลือกจากตัวอย่างภาพด้านล่าง")


def _status_html(confirmed, tentative):
    if confirmed:
        kinds = sorted({r["name_th"] for r in confirmed})
        risky = sorted({r["name_th"] for r in confirmed if r["risk"] in _HIGH_RISK})
        if risky:
            return _banner("#fdecea", "#b71c1c",
                           "🔴 พบตำหนิความเสี่ยงสูง: " + ", ".join(risky),
                           f"รวมทั้งหมด {len(kinds)} ชนิด: " + ", ".join(kinds))
        return _banner("#fff4e5", "#b26a00",
                       f"⚠️ พบตำหนิ {len(kinds)} ชนิด: " + ", ".join(kinds),
                       "ไม่มีชนิดที่จัดเป็นความเสี่ยงสูง")
    if tentative:
        kinds = sorted({r["name_th"] for r in tentative})
        return _banner("#fff8e1", "#8a6d00",
                       "🔍 อาจมีตำหนิ (ความมั่นใจต่ำ): " + ", ".join(kinds),
                       "ต่ำกว่าเกณฑ์ — แนะนำให้ตรวจซ้ำด้วยตา หรือเพิ่มโหมดความไว")
    return _banner("#e6f4ea", "#1e7e34", "✅ ไม่พบตำหนิพื้นผิว",
                   "ระบบไม่พบตำหนิในภาพนี้ (อาจเป็นภาพนอกโดเมนที่โมเดลไม่คุ้น)")


def _rows_table(rows, muted=False):
    if not rows:
        return ""
    op = "opacity:.75;" if muted else ""
    head = ("<tr style='background:#f4f6fa;text-align:left'>"
            "<th style='padding:8px 10px'>บริเวณ</th>"
            "<th style='padding:8px 10px'>ชนิดตำหนิ</th>"
            "<th style='padding:8px 10px'>ความมั่นใจ</th>"
            "<th style='padding:8px 10px'>ความเสี่ยง</th></tr>")
    body = []
    for r in rows:
        bg, fg = _RISK_COLOR.get(r["risk"], ("#eee", "#333"))
        chip = (f"<span style='background:{bg};color:{fg};padding:2px 10px;"
                f"border-radius:999px;font-size:12px;font-weight:700'>{r['risk']}</span>")
        body.append(
            "<tr style='border-top:1px solid #e6e8ec'>"
            f"<td style='padding:8px 10px'>{r['tag']}</td>"
            f"<td style='padding:8px 10px'><b>{r['name_th']}</b> "
            f"<span style='color:#8a929e;font-size:12px'>{r['class']}</span></td>"
            f"<td style='padding:8px 10px'>{r['conf']:.0%}</td>"
            f"<td style='padding:8px 10px'>{chip}</td></tr>")
    return (f"<table style='width:100%;border-collapse:collapse;font-size:14px;{op}'>"
            f"<thead>{head}</thead><tbody>{''.join(body)}</tbody></table>")


def _tentative_html(rows):
    if not rows:
        return ""
    return ("<div style='margin-top:6px'><div style='font-weight:700;color:#8a6d00;"
            "margin-bottom:4px'>🔍 อาจมีตำหนิ — ความมั่นใจต่ำกว่าเกณฑ์ (ตรวจซ้ำด้วยตา)</div>"
            + _rows_table(rows, muted=True) + "</div>")


def analyze(image_rgb, conf, detailed, sensitivity, progress=gr.Progress()):
    if image_rgb is None:
        return None, None, _empty_banner(), "", "", ""

    progress(0.1, desc="โหลดโมเดล...")
    s1, s2, device = _ensure_models()
    image_bgr = _to_bgr(image_rgb)

    scale = _SENS.get(sensitivity, 1.0)
    cc = _STATE["class_conf"] or {}

    def thr(cls):
        return max(_RAW_FLOOR, cc.get(cls, conf) * scale)

    # ----- Stage 1 : หาพื้นที่ที่เป็นเหล็ก + fallback -----
    progress(0.35, desc="Stage 1: หาพื้นที่เหล็ก...")
    mask = P.run_stage1(s1, image_bgr, device)
    boxes, meta = P.build_regions(mask, image_bgr.shape)
    metal_ratio = meta["metal_ratio"]
    n_metal = meta["n_regions"] - (1 if meta["fallback_full_image"] else 0)
    stage1_img = _stage1_view(image_bgr, mask, boxes, meta)

    # ----- Stage 2 : ตรวจตำหนิ (คืนที่ conf ต่ำ แล้วมาแยกเองเป็น confirmed / tentative) -----
    progress(0.55, desc="Stage 2: ตรวจตำหนิ...")
    region_dets = []
    for x, y, w, h in boxes:
        crop = image_bgr[y:y + h, x:x + w]
        dets = P.run_stage2(s2, crop, _RAW_FLOOR, device, augment=bool(detailed),
                            class_conf=None)
        keep = []
        for d in dets:
            t = thr(d["class"])
            if d["confidence"] >= t:
                d["_status"] = "ok"
            elif d["confidence"] >= _TENTATIVE_FLOOR:
                d["_status"] = "maybe"
            else:
                continue
            cx1, cy1, cx2, cy2 = d["bbox_xyxy_crop"]
            d["bbox_xyxy_global"] = [cx1 + x, cy1 + y, cx2 + x, cy2 + y]
            keep.append(d)
        region_dets.append(keep)

    # ----- Cross-region NMS (รวม confirmed + tentative) -----
    progress(0.8, desc="รวมผล...")
    flat = [d for dets in region_dets for d in dets]
    kept_ids = {id(d) for d in P.cross_region_nms(flat, iou_thresh=0.5)}

    annotated = image_bgr.copy()
    confirmed, tentative = [], []
    for i, (x, y, w, h) in enumerate(boxes):
        dets = [d for d in region_dets[i] if id(d) in kept_ids]
        is_full = meta["fallback_full_image"] and i == len(boxes) - 1
        tag = "ทั้งภาพ" if is_full else f"#{i + 1}"
        cv2.rectangle(annotated, (x, y), (x + w, y + h),
                      (0, 165, 255) if is_full else (0, 255, 0), 2)

        ok = [d for d in dets if d["_status"] == "ok"]
        mb = [d for d in dets if d["_status"] == "maybe"]
        if ok:
            top = P.DEFECT_INFO[ok[0]["class"]]
            annotated = P.draw_thai_text(
                annotated, f"{top['name_th']} ({ok[0]['confidence']:.0%})",
                (x, y - 28), color_bgr=(0, 0, 255))
        elif mb:
            top = P.DEFECT_INFO[mb[0]["class"]]
            annotated = P.draw_thai_text(
                annotated, f"อาจเป็น {top['name_th']} ({mb[0]['confidence']:.0%})?",
                (x, y - 28), color_bgr=(0, 140, 200))
        elif not any(dd["_status"] == "ok" for reg in region_dets for dd in reg):
            annotated = P.draw_thai_text(annotated, f"เหล็ก {tag} ปกติ",
                                         (x, y - 28), color_bgr=(0, 150, 0))

        for d in dets:
            gx1, gy1, gx2, gy2 = (int(v) for v in d["bbox_xyxy_global"])
            di = P.DEFECT_INFO[d["class"]]
            row = {"tag": tag, "class": d["class"], "name_th": di["name_th"],
                   "conf": d["confidence"], "risk": di["risk"]}
            if d["_status"] == "ok":
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                confirmed.append(row)
            else:
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 140, 200), 1)
                tentative.append(row)

    confirmed.sort(key=lambda r: (_RISK_ORDER.get(r["risk"], 9), -r["conf"]))
    tentative.sort(key=lambda r: -r["conf"])

    # ----- ข้อมูลเทคนิค -----
    notes = []
    if meta["fallback_full_image"]:
        notes.append("Stage 1 เจอเหล็กน้อย (%.0f%%) จึงตรวจทั้งภาพเป็น fallback" % (metal_ratio * 100))
    if _STATE["class_conf"]:
        notes.append(f"โหมดความไว: {sensitivity} (threshold รายคลาส × {scale:g})")
    if detailed:
        notes.append("เปิดโหมดตรวจละเอียด (test-time augmentation)")
    info_md = ("`Stage 1: %d บริเวณ · เหล็กครอบคลุม %.0f%%`  `Stage 2: %d จุด (+%d อาจมี)`  `อุปกรณ์: %s`"
               % (n_metal, metal_ratio * 100, len(confirmed), len(tentative), device))
    if notes:
        info_md += "\n\n" + "\n".join("- " + n for n in notes)

    return (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), stage1_img,
            _status_html(confirmed, tentative), _rows_table(confirmed),
            _tentative_html(tentative), info_md)


def build_ui():
    sample_dir = BASE_DIR / "test_images"
    samples = [[str(p)] for p in sorted(sample_dir.glob("*"))
               if p.suffix.lower() in P.IMAGE_EXTS] if sample_dir.exists() else []

    with gr.Blocks(title="ตรวจตำหนิพื้นผิวเหล็ก") as demo:
        gr.Markdown(
            "<div class='main-title'>\n\n"
            "# 🔎 ตรวจจับตำหนิพื้นผิวเหล็ก\n"
            "ผู้ช่วยคัดกรองสภาพผิวเหล็กจากภาพถ่าย — ตรวจตำหนิ 8 ชนิด "
            "(รอยแตกลายงา · สิ่งแปลกปลอม · ผิวลอก · ผิวเป็นหลุม · สะเก็ดรีด · รอยขีดข่วน · สนิม · รอยแตกร้าว)\n"
            "<sub>prototype เพื่อการศึกษา — ไม่ใช่ระบบตรวจสอบใช้งานจริง</sub>\n\n</div>"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                inp = gr.Image(type="numpy", label="ภาพเหล็กที่จะตรวจ",
                               height=340, sources=["upload", "clipboard"])
                btn = gr.Button("ตรวจสอบอีกครั้ง", variant="primary", size="lg")
                gr.Markdown("<sub>อัปโหลด/วางภาพ แล้วระบบตรวจให้อัตโนมัติ</sub>")
                sens = gr.Radio(["มาตรฐาน", "ไว", "ไวมาก"], value="มาตรฐาน",
                                label="โหมดความไว",
                                info="ภาพถ่ายเองที่โมเดลไม่คุ้น ลองเพิ่มเป็น “ไว/ไวมาก” "
                                     "(เตือนเยอะขึ้น แต่พลาดน้อยลง)")
                if samples:
                    gr.Examples(examples=samples, inputs=inp, label="ตัวอย่างภาพ (กดเพื่อตรวจ)",
                                examples_per_page=12)
                with gr.Accordion("ตัวเลือกขั้นสูง", open=False):
                    conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                     label="Confidence ขั้นต่ำ (คลาสที่ไม่มีใน thresholds.json)")
                    detailed = gr.Checkbox(
                        value=False, label="ตรวจละเอียด (test-time augmentation)",
                        info="ช้าลง ~2–3 เท่า, recall ดีขึ้นเล็กน้อย")

            with gr.Column(scale=7):
                status = gr.HTML(_empty_banner())
                out_img = gr.Image(type="numpy",
                                   label="ผลตรวจ (แดงทึบ = ตำหนิ, ส้มบาง = อาจมี, เขียว = บริเวณเหล็ก)",
                                   height=380)
                table = gr.HTML()
                tentative = gr.HTML()
                with gr.Accordion("รายละเอียดการทำงาน (Stage 1 + เทคนิค)", open=False):
                    out_s1 = gr.Image(type="numpy",
                                      label="Stage 1 — พื้นที่ที่เป็นเหล็ก (เขียว) / fallback ทั้งภาพ (ส้ม)",
                                      height=300)
                    info = gr.Markdown()

        gr.Markdown(
            "<sub>Stage 1: DMS46 หาพื้นที่โลหะ → Stage 2: YOLO11n (เทรนบน grayscale — "
            "ระบบแปลงภาพเป็นขาวดำก่อนตรวจอัตโนมัติ). โมเดลเทรนจากชุด NEU-DET + Roboflow "
            "อาจไม่แม่นกับภาพสไตล์อื่น — ใช้โหมดความไวช่วยได้บ้าง</sub>"
        )

        outputs = [out_img, out_s1, status, table, tentative, info]
        ins = [inp, conf, detailed, sens]
        btn.click(analyze, inputs=ins, outputs=outputs)
        inp.change(analyze, inputs=ins, outputs=outputs, show_progress="minimal")
        sens.change(analyze, inputs=ins, outputs=outputs, show_progress="minimal")
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

    _CSS = ".main-title{text-align:center} footer{visibility:hidden}"
    build_ui().queue().launch(server_name=host, server_port=args.port, share=args.share,
                              theme=gr.themes.Soft(), css=_CSS)
