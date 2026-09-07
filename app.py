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
from ultralytics import YOLO

import pipeline as P
import steel_gate

try:
    import gradio as gr
except ImportError:
    sys.exit("ยังไม่ได้ติดตั้ง gradio — รัน: pip install gradio")

BASE_DIR = Path(__file__).resolve().parent

# โมเดล Stage 2 ที่เลือกได้ในหน้าเดโม — key = ป้ายในหน้าจอ, value = (weights, per-class thresholds)
# "ปรับโดเมน" (train-real1) = train-gray-n2 + 672 ภาพ corrosion จริง (RGB) — ยิงบนภาพถ่ายจริงได้จริง
# "เล่มจบ" (train-gray-n2) = grayscale, NEU benchmark — ตัวเลขในเล่ม แต่ transfer ต่ำบนภาพถ่ายจริง
_RUNS = BASE_DIR / "runs" / "detect"
_STAGE2_MODELS = {
    "ปรับโดเมน — ภาพถ่ายจริง (แนะนำ)": (_RUNS / "train-real1" / "weights" / "best.pt",
                                          BASE_DIR / "thresholds_demo.json"),
    "เล่มจบ — NEU benchmark":          (_RUNS / "train-gray-n2" / "weights" / "best.pt",
                                          BASE_DIR / "thresholds.json"),
}


def _available_models():
    """เหลือเฉพาะ key ที่ไฟล์ weights มีจริง — เรียงตามลำดับใน _STAGE2_MODELS"""
    av = {k: v for k, v in _STAGE2_MODELS.items() if v[0].exists()}
    if not av and P.STAGE2_MODEL_PATH.exists():   # fallback: default ของ pipeline.py
        av = {"ค่าเริ่มต้น (pipeline.py)": (P.STAGE2_MODEL_PATH, P.THRESHOLDS_PATH)}
    return av


_STATE = {"s1": None, "device": None, "class_conf": None, "s2_cache": {}, "cc_cache": {}}

# ระดับความเสี่ยง -> ลำดับการแสดงผล (สูงก่อน) + สีชิป (โทนอ่อน อ่านสบายตา)
_RISK_ORDER = {"สูง": 0, "ปานกลาง-สูง": 1, "ปานกลาง": 2, "ต่ำ-ปานกลาง": 3, "ต่ำ": 4}
_RISK_COLOR = {
    "สูง": ("#fef2f2", "#dc2626"),
    "ปานกลาง-สูง": ("#fff7ed", "#ea580c"),
    "ปานกลาง": ("#fefce8", "#a16207"),
    "ต่ำ-ปานกลาง": ("#eff6ff", "#2563eb"),
    "ต่ำ": ("#f1f5f9", "#475569"),
}
_HIGH_RISK = ("สูง", "ปานกลาง-สูง")

# โหมดความไว -> ตัวคูณ threshold รายคลาส (ยิ่งต่ำ = ยิ่งเตือนเยอะ)
_SENS = {"มาตรฐาน": 1.0, "ไว": 0.6, "ไวมาก": 0.4}
_RAW_FLOOR = 0.12       # conf ต่ำสุดที่ให้ YOLO คืนมา
_TENTATIVE_FLOOR = 0.30  # ต่ำกว่า threshold แต่ >= ค่านี้ = "อาจมี" (แสดงแยก)


def _load_stage2(weights_path):
    """โหลด YOLO Stage 2 + ตั้ง flag grayscale จาก train_args ใน checkpoint (เหมือน pipeline.load_models)"""
    m = YOLO(str(weights_path))
    try:
        train_data = str((getattr(m, "ckpt", None) or {})
                         .get("train_args", {}).get("data", ""))
    except Exception:
        train_data = ""
    m._steel_gray = "gray" in train_data.lower()
    return m


def _ensure_models(model_key=None):
    """คืน (stage1, stage2, device). stage2 เลือกได้ตาม model_key — cache ไว้ทุกตัวที่เคยโหลด"""
    if _STATE["s1"] is None:
        _STATE["device"] = P.resolve_device("auto")
        print("  Stage 1 (Metal Localization / DMS46)...")
        import torch
        s1 = torch.jit.load(str(P.STAGE1_MODEL_PATH), map_location=_STATE["device"])
        s1.eval()
        if _STATE["device"] == "cuda":
            s1 = s1.cuda()
        _STATE["s1"] = s1

    av = _available_models()
    if model_key not in av:
        model_key = next(iter(av))                # default = ตัวแรก (train-real1 ถ้ามี)
    _STATE["s2_key"] = model_key
    weights, thr_path = av[model_key]
    key = str(weights)
    if key not in _STATE["s2_cache"]:
        print(f"  Stage 2 ({model_key}): {weights.relative_to(BASE_DIR)}  thr={thr_path.name}")
        _STATE["s2_cache"][key] = _load_stage2(weights)
        _STATE["cc_cache"][key] = P.load_class_conf(thr_path) or {}
    _STATE["class_conf"] = _STATE["cc_cache"][key]
    return _STATE["s1"], _STATE["s2_cache"][key], _STATE["device"]


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
    lw = max(2, round(max(view.shape[:2]) / 400))
    for i, (x, y, w, h) in enumerate(boxes):
        is_full = meta["fallback_full_image"] and i == len(boxes) - 1
        col = (0, 165, 255) if is_full else (0, 255, 0)  # ส้ม = fallback ทั้งภาพ
        cv2.rectangle(view, (x, y), (x + w, y + h), col, lw)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


# ---------- HTML rendering ----------
# state -> สีเส้นขอบซ้ายของการ์ดสรุปผล (โทนเดียวกับ _RISK_COLOR — เรียบ ไม่มีพื้นสีจัด)
_STATE_ACCENT = {
    "danger": "#ef4444", "warn": "#f59e0b", "maybe": "#94a3b8",
    "ok": "#22c55e", "neutral": "#cbd5e1",
}


def _card(state, title, sub=""):
    accent = _STATE_ACCENT.get(state, _STATE_ACCENT["neutral"])
    sub_html = f"<div class='rc-sub'>{sub}</div>" if sub else ""
    return (f"<div class='rc' style='border-left-color:{accent}'>"
            f"<div class='rc-kicker'>ผลการตรวจ</div>"
            f"<div class='rc-title'>{title}</div>{sub_html}</div>")


def _empty_banner():
    return _card("neutral", "ยังไม่มีผล",
                 "อัปโหลดหรือวางภาพเหล็ก ระบบจะตรวจให้อัตโนมัติ — หรือเลือกจากภาพตัวอย่างด้านล่าง")


def _status_html(confirmed, tentative):
    if confirmed:
        kinds = sorted({r["name_th"] for r in confirmed})
        risky = sorted({r["name_th"] for r in confirmed if r["risk"] in _HIGH_RISK})
        extra = f" · อาจมีเพิ่มอีก {len(tentative)} จุด" if tentative else ""
        if risky:
            return _card("danger", "พบตำหนิความเสี่ยงสูง",
                         ", ".join(risky) + f"  (รวมทั้งหมด {len(kinds)} ชนิด){extra}")
        return _card("warn", f"พบตำหนิ {len(kinds)} ชนิด",
                     ", ".join(kinds) + "  ไม่มีชนิดที่จัดเป็นความเสี่ยงสูง" + extra)
    if tentative:
        kinds = sorted({r["name_th"] for r in tentative})
        return _card("maybe", "อาจมีตำหนิ (ความมั่นใจต่ำ)",
                     ", ".join(kinds) + "  ต่ำกว่าเกณฑ์ — แนะนำให้ตรวจซ้ำด้วยตา หรือปรับโหมดความไวขึ้น")
    return _card("ok", "ไม่พบตำหนิพื้นผิว",
                 "ระบบไม่พบตำหนิในภาพนี้ (ถ้าเป็นภาพสไตล์ที่โมเดลไม่คุ้น อาจพลาดได้)")


def _rows_table(confirmed, tentative):
    rows = ([dict(r, _muted=False) for r in confirmed]
            + [dict(r, _muted=True) for r in tentative])
    if not rows:
        return ""
    head = ("<tr><th>บริเวณ</th><th>ชนิดตำหนิ</th>"
            "<th>ความมั่นใจ</th><th>ความเสี่ยง</th></tr>")
    body = []
    for r in rows:
        bg, fg = _RISK_COLOR.get(r["risk"], ("#eef0f3", "#586070"))
        chip = (f"<span class='rt-chip' style='background:{bg};color:{fg}'>{r['risk']}</span>")
        conf = f"{r['conf']:.0%}" + ("  <span class='rt-tag'>ต่ำกว่าเกณฑ์</span>" if r["_muted"] else "")
        body.append(
            f"<tr class='{'rt-muted' if r['_muted'] else ''}'>"
            f"<td>{r['tag']}</td>"
            f"<td><span class='rt-name'>{r['name_th']}</span>"
            f"<span class='rt-cls'>{r['class']}</span></td>"
            f"<td>{conf}</td><td>{chip}</td></tr>")
    return (f"<table class='rt'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def analyze(image_rgb, conf, detailed, sensitivity, model_key, gate_on, progress=gr.Progress()):
    try:
        return _analyze(image_rgb, conf, detailed, sensitivity, model_key, gate_on, progress)
    except Exception as e:                       # เดโมต้องไม่ค้าง — โชว์ error เป็นการ์ดแทน stack trace
        import traceback
        traceback.print_exc()
        return (None, None,
                _card("danger", "ประมวลผลภาพนี้ไม่สำเร็จ", str(e)),
                "", f"`{type(e).__name__}: {e}`")


def _analyze(image_rgb, conf, detailed, sensitivity, model_key, gate_on, progress):
    if image_rgb is None:
        return None, None, _empty_banner(), "", ""

    image_bgr = _to_bgr(image_rgb)

    # Stage 0: P(พื้นผิวเหล็ก) — ใช้เป็น "เสียงโหวตเดียว" ตอนสรุปผล ไม่บล็อกเดี่ยว ๆ
    # (classifier ยัง overfit ไป lab crop — real steel photo ได้ P ต่ำ จึงไม่ยอมให้มัน veto)
    gate_p = steel_gate.steel_prob(image_bgr) if gate_on else None

    progress(0.1, desc="โหลดโมเดล...")
    s1, s2, device = _ensure_models(model_key)

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
    # ปรับความหนาเส้น/ขนาดฟอนต์ตามขนาดภาพ — ภาพใหญ่ต้องเส้นหนา ตัวใหญ่ ถึงจะเห็นชัด
    S = max(annotated.shape[:2])
    lw = max(3, round(S / 400))                 # กรอบตำหนิที่ยืนยัน
    lw_thin = max(2, lw - 2)                    # กรอบ "อาจมี"
    fpx = int(min(72, max(22, S / 34)))         # ฟอนต์ป้ายกำกับ
    C_OK, C_MAYBE, C_REGION = (68, 68, 239), (11, 158, 245), (94, 197, 34)  # BGR ~ #ef4444 / #f59e0b / #22c55e

    confirmed, tentative = [], []
    for i, (x, y, w, h) in enumerate(boxes):
        dets = [d for d in region_dets[i] if id(d) in kept_ids]
        is_full = meta["fallback_full_image"] and i == len(boxes) - 1
        tag = "ทั้งภาพ" if is_full else f"#{i + 1}"
        # วาดกรอบบริเวณเหล็กเฉพาะตอนมีหลายบริเวณจริง — ภาพทั่วไป/fallback ไม่ต้องรก
        if not is_full and len(boxes) > 1:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), C_REGION, lw_thin)

        ok = [d for d in dets if d["_status"] == "ok"]
        mb = [d for d in dets if d["_status"] == "maybe"]
        ly = max(y - fpx - 8, 4)
        if ok:
            top = P.DEFECT_INFO[ok[0]["class"]]
            annotated = P.draw_thai_text(
                annotated, f"{top['name_th']} ({ok[0]['confidence']:.0%})",
                (x, ly), color_bgr=C_OK, font_size=fpx)
        elif mb:
            top = P.DEFECT_INFO[mb[0]["class"]]
            annotated = P.draw_thai_text(
                annotated, f"อาจเป็น {top['name_th']} ({mb[0]['confidence']:.0%})?",
                (x, ly), color_bgr=C_MAYBE, font_size=fpx)
        elif not any(dd["_status"] == "ok" for reg in region_dets for dd in reg):
            annotated = P.draw_thai_text(annotated, f"เหล็ก {tag} ปกติ",
                                         (x, ly), color_bgr=(0, 150, 0), font_size=fpx)

        for d in dets:
            gx1, gy1, gx2, gy2 = (int(v) for v in d["bbox_xyxy_global"])
            di = P.DEFECT_INFO[d["class"]]
            row = {"tag": tag, "class": d["class"], "name_th": di["name_th"],
                   "conf": d["confidence"], "risk": di["risk"]}
            if d["_status"] == "ok":
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 255, 255), lw + 2)
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), C_OK, lw)
                confirmed.append(row)
            else:
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), C_MAYBE, lw_thin)
                tentative.append(row)

    confirmed.sort(key=lambda r: (_RISK_ORDER.get(r["risk"], 9), -r["conf"]))
    tentative.sort(key=lambda r: -r["conf"])

    # ----- ข้อมูลเทคนิค -----
    notes = [f"โมเดล Stage 2: {_STATE.get('s2_key', model_key)}"
             + ("  (แปลง crop เป็นขาวดำก่อนตรวจ)" if getattr(s2, "_steel_gray", False) else "")]
    if gate_p is not None:
        notes.append(f"Stage 0: ตัวจำแนกพื้นผิวประเมิน P(เหล็ก) = {gate_p:.0%}"
                     + ("  (ต่ำ — classifier ยัง bias ไปภาพแล็บ ใช้ประกอบเท่านั้น)"
                        if gate_p < 0.5 else ""))
    elif gate_on:
        notes.append("Stage 0: ยังไม่มีโมเดล gate (รัน train_gate.py) — ข้ามการเช็คพื้นผิวเหล็ก")
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

    # "ไม่พบพื้นผิวเหล็ก" — เชื่อได้เฉพาะตอนทุกสัญญาณเงียบพร้อมกัน:
    #   ตัวจำแนกพื้นผิว P(เหล็ก) ต่ำมาก  +  DMS46 เจอโลหะ ~0%  +  Stage 2 ไม่เจอตำหนิเลย
    # (classifier ยัง bias ไปภาพแล็บ จึงไม่ให้มัน veto detection ที่ Stage 2 มั่นใจ)
    no_steel = (gate_p is not None and gate_p < 0.10
                and metal_ratio < 0.02 and not confirmed and not tentative)
    if no_steel:
        status = _card("neutral", "ไม่พบพื้นผิวเหล็กในภาพนี้",
                       f"ทั้ง Stage 1, Stage 2 และตัวจำแนกพื้นผิว ({gate_p:.0%}) "
                       f"เห็นตรงกันว่าภาพนี้ไม่ใช่พื้นผิวเหล็ก")
    else:
        status = _status_html(confirmed, tentative)

    return (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), stage1_img,
            status, _rows_table(confirmed, tentative), info_md)


def _globs(d):
    return [[str(p)] for p in sorted(d.glob("*"))
            if p.suffix.lower() in P.IMAGE_EXTS] if d.exists() else []


_CSS = """
:root{color-scheme:light}
footer{display:none!important}
body,.gradio-container{background:#ffffff!important}
.gradio-container{max-width:1200px!important;margin:0 auto!important}
/* ภาพผลตรวจ — ใหญ่ สว่าง เห็นตำหนิชัด */
.result-img{border:1px solid #eef1f4;border-radius:12px;background:#fbfcfd}
.result-img img{object-fit:contain!important}
.hd{padding:8px 2px 16px}
.hd-title{font-size:23px;font-weight:700;color:#0f172a;letter-spacing:.2px}
.hd-sub{font-size:13.5px;color:#64748b;margin-top:6px;line-height:1.55}
.hd-note{font-size:11.5px;color:#aeb7c2;margin-top:4px}
.hint{font-size:12px;color:#94a3b8;margin:-2px 0 8px}
/* การ์ดสรุปผล */
.rc{border:1px solid #eef1f4;border-left:5px solid #cbd5e1;border-radius:12px;
    padding:14px 18px;background:#fff;box-shadow:0 1px 3px rgba(15,23,42,.05)}
.rc-kicker{font-size:10.5px;letter-spacing:.12em;color:#aeb7c2;text-transform:uppercase}
.rc-title{font-size:17.5px;font-weight:700;color:#0f172a;line-height:1.35;margin-top:3px}
.rc-sub{font-size:13px;color:#5b6675;margin-top:6px;line-height:1.55}
/* legend ใต้ภาพผล */
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:10px 2px 2px}
.lg{font-size:11.5px;color:#64748b;display:flex;align-items:center}
.lg::before{content:"";width:12px;height:12px;border-radius:3px;margin-right:6px}
.lg-def::before{background:#ef4444}
.lg-may::before{background:#f59e0b}
.lg-metal::before{background:#fff;border:2px solid #22c55e}
/* ตารางผล */
table.rt{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:10px;
    background:#fff;border:1px solid #eef1f4;border-radius:12px;overflow:hidden}
table.rt th{text-align:left;font-weight:600;color:#8a94a3;font-size:11px;
    letter-spacing:.05em;text-transform:uppercase;padding:9px 12px;background:#f8fafc;
    border-bottom:1px solid #eef1f4}
table.rt td{padding:11px 12px;border-bottom:1px solid #f2f5f8;color:#1f2937;vertical-align:middle}
table.rt tr:last-child td{border-bottom:none}
.rt-name{font-weight:600;color:#0f172a}
.rt-cls{color:#aeb7c2;font-size:11.5px;margin-left:7px}
.rt-chip{padding:2px 10px;border-radius:999px;font-size:11.5px;font-weight:600;white-space:nowrap}
.rt-tag{color:#aeb7c2;font-size:11px}
tr.rt-muted td{color:#94a3b8;background:#fcfdfe}
tr.rt-muted .rt-name{font-weight:500;color:#64748b}
.foot{font-size:11.5px;color:#a3adba;line-height:1.65;padding:16px 2px 2px;
    border-top:1px solid #f2f5f8;margin-top:18px}
"""


def build_ui():
    lab_samples = _globs(BASE_DIR / "test_images")                 # NEU/Rust crop — โชว์ครบ 8 คลาส
    real_samples = _globs(BASE_DIR / "real_test" / "images")       # ภาพถ่ายจริงระดับ scene
    model_choices = list(_available_models())

    with gr.Blocks(title="ตรวจตำหนิพื้นผิวเหล็ก",
                   theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray"),
                   css=_CSS) as demo:
        gr.HTML(
            "<div class='hd'>"
            "<div class='hd-title'>ตรวจจับตำหนิพื้นผิวเหล็ก</div>"
            "<div class='hd-sub'>ผู้ช่วยคัดกรองสภาพผิวเหล็กจากภาพถ่าย · ตรวจตำหนิ 8 ชนิด "
            "(รอยแตกลายงา, สิ่งแปลกปลอม, ผิวลอก, ผิวเป็นหลุม, สะเก็ดรีด, รอยขีดข่วน, สนิม, รอยแตกร้าว)</div>"
            "<div class='hd-note'>prototype เพื่อการศึกษา — ไม่ใช่ระบบตรวจสอบใช้งานจริง</div>"
            "</div>"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=300):
                inp = gr.Image(type="numpy", label="ภาพเหล็กที่จะตรวจ",
                               height=300, sources=["upload", "clipboard"])
                gr.HTML("<div class='hint'>อัปโหลดหรือวางภาพ แล้วระบบตรวจให้อัตโนมัติ</div>")
                sens = gr.Radio(["มาตรฐาน", "ไว", "ไวมาก"], value="มาตรฐาน",
                                label="โหมดความไว",
                                info="ภาพที่โมเดลไม่คุ้น เพิ่มเป็น ไว / ไวมาก ได้ "
                                     "(เตือนมากขึ้น แต่พลาดน้อยลง)")
                with gr.Accordion("ตัวเลือกขั้นสูง", open=False):
                    model_sel = gr.Radio(
                        model_choices, value=model_choices[0] if model_choices else None,
                        label="โมเดล Stage 2",
                        info="ปรับโดเมน = เทรนเพิ่มด้วยภาพถ่ายจริง (ยิงบนภาพถ่ายจริงได้ดีกว่า) · "
                             "เล่มจบ = ตัวที่รายงานตัวเลขในเล่ม (grayscale, NEU benchmark)")
                    conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                     label="Confidence ขั้นต่ำ (คลาสที่ไม่มีในไฟล์ threshold)")
                    detailed = gr.Checkbox(
                        value=False, label="ตรวจละเอียด (test-time augmentation)",
                        info="ช้าลงราว 2–3 เท่า แลกกับ recall ที่ดีขึ้นเล็กน้อย")
                    gate_on = gr.Checkbox(
                        value=steel_gate.available(), interactive=steel_gate.available(),
                        label="Stage 0: ตัวจำแนกพื้นผิวเหล็ก",
                        info=("ช่วยตอบ ไม่พบพื้นผิวเหล็ก เมื่อ Stage 1/2 และตัวจำแนกเห็นตรงกันว่าไม่ใช่เหล็ก"
                              if steel_gate.available()
                              else "ยังไม่มีโมเดล gate — รัน train_gate.py ก่อน"))
                    btn = gr.Button("ประมวลผลใหม่", variant="secondary", size="sm")

            with gr.Column(scale=6, min_width=340):
                status = gr.HTML(_empty_banner())

        out_img = gr.Image(type="numpy", label="ผลตรวจ", height=620,
                           interactive=False, elem_classes=["result-img"])
        gr.HTML("<div class='legend'>"
                "<span class='lg lg-def'>กรอบแดง = ตำหนิที่ยืนยัน</span>"
                "<span class='lg lg-may'>กรอบส้ม = อาจมี</span>"
                "<span class='lg lg-metal'>กรอบเขียว = บริเวณที่เป็นเหล็ก (เฉพาะเมื่อมีหลายบริเวณ)</span>"
                "</div>")
        table = gr.HTML()
        with gr.Accordion("การทำงานภายใน (Stage 1 + ข้อมูลเทคนิค)", open=False):
            out_s1 = gr.Image(type="numpy",
                              label="Stage 1 — บริเวณที่เป็นเหล็ก (เขียว) / ตรวจทั้งภาพ (ส้ม)",
                              height=440)
            info = gr.Markdown()

        gr.HTML(
            "<div class='foot'>Stage 1: DMS46 หาพื้นที่โลหะ (soft-gate + ตรวจทั้งภาพเมื่อไม่พบ) "
            "จากนั้น Stage 2: YOLO11n ตรวจตำหนิ 8 ชนิด. โมเดลเทรนจาก NEU-DET + Roboflow "
            "(บวกภาพถ่ายจริงสำหรับตัว ปรับโดเมน). ภาพสไตล์อื่นอาจพลาด — เพิ่มโหมดความไว "
            "หรือสลับโมเดลใน ตัวเลือกขั้นสูง</div>"
        )

        outputs = [out_img, out_s1, status, table, info]
        ins = [inp, conf, detailed, sens, model_sel, gate_on]
        btn.click(analyze, inputs=ins, outputs=outputs)
        inp.change(analyze, inputs=ins, outputs=outputs, show_progress="minimal")
        sens.change(analyze, inputs=ins, outputs=outputs, show_progress="minimal")
        model_sel.change(analyze, inputs=ins, outputs=outputs, show_progress="minimal")
        gate_on.change(analyze, inputs=ins, outputs=outputs, show_progress="minimal")

        if real_samples:
            gr.Examples(examples=real_samples, inputs=inp, label="ภาพถ่ายจริง",
                        examples_per_page=12)
        if lab_samples:
            gr.Examples(examples=lab_samples, inputs=inp,
                        label="ภาพตัวอย่างชุด benchmark (ครบ 8 คลาส)", examples_per_page=12)
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

    av = _available_models()
    if not av:
        sys.exit("ไม่พบไฟล์ weights ของ Stage 2 เลย — เทรนก่อนด้วย train.py / run_round.py")
    print("กำลังเตรียมโมเดล (โหลดครั้งเดียวตอนเริ่ม)...")
    for k in av:                     # warm ทุกโมเดลที่เลือกได้ ให้สลับในหน้าเดโมแล้วไม่ต้องรอโหลด
        _ensure_models(k)
    _ensure_models(next(iter(av)))   # ให้ default เป็นตัวที่ active ล่าสุด
    print(f"โมเดล Stage 2 ที่ใช้ได้: {', '.join(av)}  (ค่าเริ่มต้น: {next(iter(av))})\n")

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
