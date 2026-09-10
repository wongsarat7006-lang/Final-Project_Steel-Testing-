"""
Prototype UI (Gradio): อัปโหลด / วางภาพ / ถ่ายจากกล้อง -> ระบบตรวจ 2 ขั้นตอน
-> การ์ดสรุปผล + ตารางรายการตำหนิ + ภาพผลลัพธ์ + ภาพ Stage 1

รัน:
    python app.py                 # เปิดที่ http://127.0.0.1:7860 (และวง LAN เดียวกัน)
    python app.py --share         # เพิ่มลิงก์สาธารณะ *.gradio.live
    python app.py --local-only    # เปิดเฉพาะเครื่องนี้

ต้องมี gradio:  pip install gradio
หมายเหตุ: การถ่ายจากกล้องในเบราว์เซอร์ต้องเปิดผ่าน https หรือ localhost (127.0.0.1)
"""
import csv
import json
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import pipeline as P
import steel_gate

LOG_DIR = Path(__file__).resolve().parent / "demo_logs"

try:
    import gradio as gr
except ImportError:
    sys.exit("ยังไม่ได้ติดตั้ง gradio — รัน: pip install gradio")

BASE_DIR = Path(__file__).resolve().parent

# โมเดล Stage 2 ที่เลือกได้ในหน้าเดโม — key = ป้ายในหน้าจอ, value = (weights, per-class thresholds)
# "ปรับโดเมน v3" (train-real3) = train-real1 data + 683 ภาพ corrosion จริง (round3: เสาส่งไฟ/ท่อ/แผ่นเหล็ก)
#   real_test: rust 8/12 -> 12/12 (P 0.80 -> 0.92), micro-F1 0.51 -> 0.68 · lab mAP50 0.828 (rust 0.995)
# "ปรับโดเมน v1" (train-real1) = train-gray-n2 config + 672 ภาพ corrosion จริง (round1) — สนิม lab-crop conf ~0.42, ภาพจริง 8/12
# "รุ่นทดลอง scene" (train-real2) = + 1022 ภาพสนิม scene จริง (round2) — REGRESS บนสนิม lab-crop (fine-tune แรง + เทรนไม่จบ)
# "เล่มจบ" (train-gray-n2) = grayscale, NEU benchmark — ตัวเลขในเล่ม แต่ transfer ต่ำบนภาพถ่ายจริง
_RUNS = BASE_DIR / "runs" / "detect"
_STAGE2_MODELS = {
    "ปรับโดเมน v3 — ภาพถ่ายจริง (แนะนำ)": (_RUNS / "train-real3" / "weights" / "best.pt",
                                             BASE_DIR / "thresholds_real3.json"),
    "ปรับโดเมน v1 (round 1)":            (_RUNS / "train-real1" / "weights" / "best.pt",
                                          BASE_DIR / "thresholds_demo.json"),
    "รุ่นทดลอง — เน้นภาพ scene (round 2)": (_RUNS / "train-real2" / "weights" / "best.pt",
                                              BASE_DIR / "thresholds_real2.json"),
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


def _status_html(confirmed, tentative, maybe_not_steel=False, gate_p=None):
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
    # ไม่พบตำหนิ — ถ้าตัวจำแนกพื้นผิว + Stage 1 เงียบพร้อมกัน แปะหมายเหตุ (ไม่ฟันธง:
    # ทั้งคู่ bias ไปภาพแล็บ เหล็กผุ/เหล็กสนิมจริงก็ทำให้เงียบได้เหมือนกัน)
    if maybe_not_steel:
        gp = f" ({gate_p:.0%})" if gate_p is not None else ""
        return _card("ok", "ไม่พบตำหนิพื้นผิว",
                     f"ระบบไม่พบตำหนิในภาพนี้ · ตัวจำแนกพื้นผิว{gp} และ Stage 1 ประเมินว่า"
                     " ภาพนี้อาจไม่ใช่พื้นผิวเหล็ก — ถ้าเป็นเหล็กจริง แปลว่าตรวจไม่พบตำหนิ"
                     " (ภาพสไตล์ที่โมเดลไม่คุ้นอาจพลาดได้)")
    return _card("ok", "ไม่พบตำหนิพื้นผิว",
                 "ระบบไม่พบตำหนิในภาพนี้ (ถ้าเป็นภาพสไตล์ที่โมเดลไม่คุ้น อาจพลาดได้)")


TABLE_HEADERS = ["บริเวณ", "ชนิดตำหนิ", "คลาส", "ความมั่นใจ", "ความเสี่ยง"]


def _rows_data(confirmed, tentative):
    """คืนข้อมูลตารางเป็น list-of-rows สำหรับ gr.Dataframe (ยืนยันก่อน แล้วตามด้วย 'อาจมี')"""
    out = []
    for r in confirmed:
        out.append([r["tag"], r["name_th"], r["class"], f"{r['conf']:.0%}", r["risk"]])
    for r in tentative:
        out.append([r["tag"], r["name_th"], r["class"],
                    f"{r['conf']:.0%} (อาจมี)", r["risk"]])
    return out


def _causes_html(confirmed, tentative):
    """กล่องข้อมูลอ้างอิง: สาเหตุที่พบบ่อย + คำแนะนำ ต่อชนิดตำหนิที่ตรวจเจอ (รวม 'อาจมี')"""
    seen, items = set(), []
    for r in list(confirmed) + list(tentative):
        cls = r["class"]
        if cls in seen:
            continue
        seen.add(cls)
        di = P.DEFECT_INFO.get(cls, {})
        if not di.get("causes"):
            continue
        items.append(
            "<div class='causes-item'>"
            f"<b>{di['name_th']}</b><span class='causes-risk'>ความเสี่ยง: {di['risk']}</span>"
            f"<div>สาเหตุที่พบบ่อย: {di['causes']}</div>"
            f"<div class='causes-adv'>คำแนะนำ: {di['advice']}</div>"
            "</div>")
    if not items:
        return ""
    return ("<div class='causes'>"
            "<div class='causes-h'>สาเหตุที่อาจทำให้เกิดตำหนิเหล่านี้ + คำแนะนำ"
            "<span>ข้อมูลอ้างอิงทั่วไปของตำหนิแต่ละชนิด — ไม่ใช่การวินิจฉัยชิ้นงานในภาพนี้</span></div>"
            + "".join(items) + "</div>")


def analyze(image_rgb, conf, detailed, sensitivity, model_key, gate_on, multiscale=False,
            progress=gr.Progress()):
    try:
        return _analyze(image_rgb, conf, detailed, sensitivity, model_key, gate_on,
                        multiscale, progress)
    except Exception as e:                       # เดโมต้องไม่ค้าง — โชว์ error เป็นการ์ดแทน stack trace
        import traceback
        traceback.print_exc()
        return (None,
                _card("danger", "ประมวลผลภาพนี้ไม่สำเร็จ", str(e)),
                [], "", {})


def _analyze(image_rgb, conf, detailed, sensitivity, model_key, gate_on, multiscale, progress):
    if image_rgb is None:
        return None, _empty_banner(), [], "", {}

    image_bgr = _to_bgr(image_rgb)

    # Stage 0: P(พื้นผิวเหล็ก) — ใช้เป็น "เสียงโหวตเดียว" ตอนสรุปผล ไม่บล็อกเดี่ยว ๆ
    # (classifier ยัง overfit ไป lab crop — real steel photo ได้ P ต่ำ จึงไม่ยอมให้มัน veto)
    gate_p = steel_gate.steel_prob(image_bgr) if gate_on else None

    progress(0.1, desc="โหลดโมเดล...")
    s1, s2, device = _ensure_models(model_key)

    scale = _SENS.get(sensitivity, 1.0)
    cc = _STATE["class_conf"] or {}

    # โหมดความไวคุมทั้ง threshold + ความละเอียดการตรวจ (ผู้ทดสอบเลือกจุดเดียว)
    #   ไว     -> ลด threshold + เปิด TTA (พลิก/ย่อ-ขยาย) — ช้าขึ้น ~2-3 เท่า
    #   ไวมาก  -> + ตรวจหลายสเกล/ตัดไทล์ — ช้าขึ้น ~3-6 เท่า เจอบนภาพใหญ่/ภาพจริงมากขึ้น
    use_tta = bool(detailed) or sensitivity in ("ไว", "ไวมาก")
    use_ms = bool(multiscale) or sensitivity == "ไวมาก"

    def thr(cls):                       # threshold ที่ลดตามโหมดความไว
        return max(_RAW_FLOOR, cc.get(cls, conf) * scale)

    def thr_tuned(cls):                 # threshold ที่จูนไว้ (เทียบเท่าโหมด "มาตรฐาน")
        return max(_RAW_FLOOR, cc.get(cls, conf))

    # ----- Stage 1 : หาพื้นที่ที่เป็นเหล็ก + fallback -----
    progress(0.35, desc="Stage 1: หาพื้นที่เหล็ก...")
    mask = P.run_stage1(s1, image_bgr, device)
    boxes, meta = P.build_regions(mask, image_bgr.shape)
    metal_ratio = meta["metal_ratio"]

    # ภาพ scene/มุมกว้าง: Stage 1 มักแตกเหล็กเป็นหลายชิ้นเล็ก ๆ ทำให้ Stage 2 เสียบริบท
    # -> ถ้าแตกเป็นหลายบริเวณจริง (>=3) เพิ่ม "ตรวจทั้งภาพ" อีก 1 รอบ แล้วให้ NMS รวมผลเอง
    #    (detection จากรอบทั้งภาพจะถูกกรองด้วย threshold ที่เข้มกว่า เพราะภาพถูกย่อมาก)
    H, W = image_bgr.shape[:2]
    full_box = (0, 0, W, H)
    add_full = (not meta["fallback_full_image"] and len(boxes) >= 3 and full_box not in boxes)
    if add_full:
        boxes = list(boxes) + [full_box]

    # ----- Stage 2 : ตรวจตำหนิ (คืนที่ conf ต่ำ แล้วมาแยกเองเป็น confirmed / tentative) -----
    progress(0.55, desc=("Stage 2: ตรวจหลายสเกล..." if use_ms else "Stage 2: ตรวจตำหนิ..."))
    region_dets = []
    for x, y, w, h in boxes:
        is_full_pass = (x, y, w, h) == full_box and add_full
        k = 1.6 if is_full_pass else 1.0        # รอบ "ทั้งภาพ" ภาพถูกย่อมาก -> ต้องมั่นใจกว่าถึงจะนับ
        crop = image_bgr[y:y + h, x:x + w]
        if use_ms:
            dets = P.run_stage2_multiscale(s2, crop, _RAW_FLOOR, device, class_conf=None)
        else:
            dets = P.run_stage2(s2, crop, _RAW_FLOOR, device, augment=use_tta,
                                class_conf=None)
        keep = []
        for d in dets:
            t = thr(d["class"]) * k
            t_tuned = thr_tuned(d["class"]) * k
            # ยืนยัน ("ok") เฉพาะที่ผ่าน threshold ที่จูนไว้ — detection ที่ผ่านได้เพราะ
            # โหมดความไวลด threshold ลง จะแสดงเป็น "อาจมี" เท่านั้น ไม่ขึ้น "ความเสี่ยงสูง"
            # (กันเคสเช่น ผนังปูน conf 0.19 กลายเป็น "พบตำหนิความเสี่ยงสูง")
            if d["confidence"] >= t_tuned:
                d["_status"] = "ok"
            elif d["confidence"] >= min(_TENTATIVE_FLOOR * k, t):
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
    C_OK, C_MAYBE = (68, 68, 239), (11, 158, 245)  # BGR ~ #ef4444 / #f59e0b

    confirmed, tentative = [], []
    for i, (x, y, w, h) in enumerate(boxes):
        dets = [d for d in region_dets[i] if id(d) in kept_ids]
        is_full = (x, y, w, h) == full_box
        tag = "ทั้งภาพ" if is_full else f"#{i + 1}"
        # กรอบบริเวณเหล็ก (เขียว) แสดงเฉพาะในภาพ Stage 1 เท่านั้น — ภาพผลลัพธ์ให้เห็นแค่กรอบตำหนิ

        ok = [d for d in dets if d["_status"] == "ok"]
        mb = [d for d in dets if d["_status"] == "maybe"]
        # ป้ายกำกับ: ปกติวางเหนือกรอบ แต่ถ้าไม่มีที่ (กรอบชิดขอบบน / เต็มภาพ) ให้วางในกรอบ เยื้องเข้ามาเล็กน้อย
        lx = min(max(x + 4, 4), annotated.shape[1] - 6)
        ly = y - fpx - 10 if y - fpx - 10 >= 2 else y + 6
        if ok:
            top = P.DEFECT_INFO[ok[0]["class"]]
            annotated = P.draw_thai_text(
                annotated, f"{top['name_th']} ({ok[0]['confidence']:.0%})",
                (lx, ly), color_bgr=C_OK, font_size=fpx)
        elif mb:
            top = P.DEFECT_INFO[mb[0]["class"]]
            annotated = P.draw_thai_text(
                annotated, f"อาจเป็น {top['name_th']} ({mb[0]['confidence']:.0%})?",
                (lx, ly), color_bgr=C_MAYBE, font_size=fpx)

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

    # "อาจไม่ใช่พื้นผิวเหล็ก" — เป็นแค่หมายเหตุ ไม่ใช่คำตัดสิน:
    #   ตัวจำแนกพื้นผิว + DMS46 ทั้งคู่ bias ไปภาพแล็บ crop ระยะใกล้ → เหล็กสนิม/เหล็กผุ
    #   ในภาพถ่ายจริงก็ได้ P(เหล็ก)≈0 และ metal_ratio≈0 เหมือนกัน (เจอกับ real_test หลายภาพ)
    #   เดิมโชว์เป็น headline "ไม่พบพื้นผิวเหล็ก" → ฟันธงผิดกับภาพเหล็กจริงที่ยากที่สุด
    maybe_not_steel = (gate_p is not None and gate_p < 0.10
                       and metal_ratio < 0.02 and not confirmed and not tentative)
    status = _status_html(confirmed, tentative, maybe_not_steel, gate_p)

    rows = _rows_data(confirmed, tentative)
    causes_html = _causes_html(confirmed, tentative)
    verdict = re.sub(r"<[^>]+>", " ", status)
    verdict = re.sub(r"\s+", " ", verdict).strip()
    state = {
        "model": _STATE.get("s2_key", model_key),
        "verdict": verdict,
        "sensitivity": sensitivity,
        "gate_p": round(gate_p, 3) if gate_p is not None else None,
        "stage1_metal_ratio": round(metal_ratio, 4),
        "fallback_full_image": meta["fallback_full_image"],
        "detections": [
            {"region": r["tag"], "class": r["class"], "name_th": r["name_th"],
             "confidence": round(r["conf"], 4), "risk": r["risk"], "status": "confirmed"}
            for r in confirmed
        ] + [
            {"region": r["tag"], "class": r["class"], "name_th": r["name_th"],
             "confidence": round(r["conf"], 4), "risk": r["risk"], "status": "tentative"}
            for r in tentative
        ],
    }
    return (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            status, rows, causes_html, state)


def _globs(d):
    return [[str(p)] for p in sorted(d.glob("*"))
            if p.suffix.lower() in P.IMAGE_EXTS] if d.exists() else []


def _globs_rec(d):
    return [[str(p)] for p in sorted(d.rglob("*"))
            if p.suffix.lower() in P.IMAGE_EXTS] if d.exists() else []


# ---------- ดาวน์โหลดผล + feedback ----------
def prepare_download(annotated_rgb, state):
    """เขียนผลลัพธ์ (ภาพ annotated + JSON) เป็นไฟล์ zip ให้กดดาวน์โหลด"""
    if annotated_rgb is None or not state:
        return gr.update(visible=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = Path(tempfile.mkdtemp(prefix="steeldemo_"))
    jpg = tmp / f"result_{ts}.jpg"
    cv2.imwrite(str(jpg), cv2.cvtColor(np.asarray(annotated_rgb), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    js = tmp / f"result_{ts}.json"
    js.write_text(json.dumps({"timestamp": ts, **state}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    zp = tmp / f"steel_result_{ts}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jpg, jpg.name)
        z.write(js, js.name)
    return gr.update(value=str(zp), visible=True)


def submit_feedback(orig_rgb, annotated_rgb, state, rating, comment):
    """บันทึก feedback ของผู้ทดลอง -> demo_logs/feedback.csv (+ ภาพ/ผล ถ้ามี rating)"""
    if not rating:
        return "เลือกระดับผลตรวจก่อนกดส่ง"
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    row_dir = LOG_DIR / ts
    row_dir.mkdir(parents=True, exist_ok=True)
    try:
        if orig_rgb is not None:
            cv2.imwrite(str(row_dir / "input.jpg"),
                        cv2.cvtColor(np.asarray(orig_rgb), cv2.COLOR_RGB2BGR))
        if annotated_rgb is not None:
            cv2.imwrite(str(row_dir / "result.jpg"),
                        cv2.cvtColor(np.asarray(annotated_rgb), cv2.COLOR_RGB2BGR))
        (row_dir / "result.json").write_text(
            json.dumps(state or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print("feedback save (image) error:", e)
    csv_path = LOG_DIR / "feedback.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "rating", "comment", "model", "verdict",
                        "n_detections", "folder"])
        st = state or {}
        w.writerow([ts, rating, (comment or "").replace("\n", " ").strip(),
                    st.get("model", ""), st.get("verdict", ""),
                    len(st.get("detections", [])), ts])
    print(f"feedback: {rating} | {comment!r} -> {row_dir}")
    return "ขอบคุณสำหรับ feedback — บันทึกแล้ว"


_CSS = """
:root{color-scheme:light}
footer{display:none!important}
body,.gradio-container{background:#ffffff!important}
.gradio-container{max-width:1100px!important;margin:0 auto!important;padding:12px 14px 32px!important}
/* ภาพผลตรวจ — ปรับตามอัตราส่วนภาพเอง จำกัดความสูงไม่ให้ล้นจอ */
.result-img{border:1px solid #e3e8ee;border-radius:12px;background:#fbfcfd;min-height:220px}
.result-img img{object-fit:contain!important;max-height:72vh!important}
/* หัวเรื่อง */
.hd{padding:6px 2px 12px}
.hd-title{font-size:24px;font-weight:800;color:#0f172a;letter-spacing:.2px;line-height:1.25}
.hd-sub{font-size:14.5px;color:#475569;margin-top:6px;line-height:1.55}
.hd-note{font-size:12.5px;color:#94a3b8;margin-top:5px}
.hint{font-size:13px;color:#64748b;margin:-2px 0 10px;line-height:1.55}
/* การ์ดสรุปผล */
.rc{border:1px solid #e3e8ee;border-left:6px solid #cbd5e1;border-radius:12px;
    padding:16px 18px;background:#fff;box-shadow:0 1px 3px rgba(15,23,42,.06)}
.rc-kicker{font-size:11px;letter-spacing:.12em;color:#94a3b8;text-transform:uppercase;font-weight:700}
.rc-title{font-size:19px;font-weight:800;color:#0f172a;line-height:1.35;margin-top:4px}
.rc-sub{font-size:14.5px;color:#334155;margin-top:8px;line-height:1.6}
/* legend ใต้ภาพผล */
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:10px 2px 2px}
.lg{font-size:13px;color:#475569;display:flex;align-items:center}
.lg::before{content:"";width:13px;height:13px;border-radius:3px;margin-right:7px;flex:none}
.lg-def::before{background:#ef4444}
.lg-may::before{background:#f59e0b}
.foot{font-size:12.5px;color:#94a3b8;line-height:1.65;padding:16px 2px 2px;
    border-top:1px solid #eef2f6;margin-top:20px}
/* กล่องขอบเขต/ข้อจำกัด */
.limits{border:1px solid #fde3c0;background:#fff8ef;border-radius:10px;
    padding:12px 16px;margin:2px 2px 14px;font-size:13.5px;color:#7a5320;line-height:1.6}
.limits b{color:#7a3f0e}
/* กล่องสาเหตุ/คำแนะนำต่อชนิดตำหนิ */
.causes{border:1px solid #e3e8ee;background:#fbfcfe;border-radius:12px;
    padding:14px 18px;margin:12px 2px 2px;font-size:14px;color:#334155;line-height:1.65}
.causes-h{font-size:14.5px;font-weight:800;color:#0f172a;margin-bottom:11px}
.causes-h span{display:block;font-size:12px;font-weight:400;color:#94a3b8;margin-top:2px}
.causes-item{padding:11px 0;border-top:1px solid #e8edf2}
.causes-item:first-of-type{border-top:0;padding-top:2px}
.causes-item b{color:#0f172a;font-size:15px}
.causes-risk{font-size:12.5px;color:#64748b;margin-left:8px}
.causes-adv{color:#475569;margin-top:3px}
/* feedback */
.fb{border:1px solid #e3e8ee;border-radius:12px;padding:14px 18px;margin-top:12px;background:#fcfdfe}
.fb-h{font-size:14.5px;font-weight:800;color:#1e293b;margin-bottom:4px}
/* ตารางผล (gr.Dataframe) — เลื่อนแนวนอนได้เมื่อจอแคบ */
.res-table .table-wrap, .res-table table{font-size:14px!important}
.res-table{overflow-x:auto}
/* ===== จอมือถือ / จอแคบ ===== */
@media (max-width:640px){
  .gradio-container{padding:8px 10px 24px!important}
  .hd-title{font-size:20px}
  .hd-sub{font-size:13px}
  .result-img img{max-height:56vh!important}
  .rc{padding:13px 15px}
  .rc-title{font-size:17px}
  .rc-sub{font-size:13.5px}
  .causes{font-size:13.5px;padding:13px 15px}
  .res-table .table-wrap, .res-table table{font-size:13px!important}
}
"""


def build_ui():
    demo_samples = _globs_rec(BASE_DIR / "demo_samples")          # คัด 2 ภาพ/คลาส ที่โมเดลเดโมตรวจถูก
    lab_samples = _globs(BASE_DIR / "test_images")                 # NEU/Rust crop — โชว์ครบ 8 คลาส
    real_samples = _globs(BASE_DIR / "real_test" / "images")       # ภาพถ่ายจริงระดับ scene
    model_choices = list(_available_models())

    with gr.Blocks(title="ตรวจตำหนิพื้นผิวเหล็ก") as demo:
        gr.HTML(
            "<div class='hd'>"
            "<div class='hd-title'>ตรวจจับตำหนิพื้นผิวเหล็ก</div>"
            "<div class='hd-sub'>อัปโหลดหรือถ่ายภาพผิวเหล็ก ระบบจะคัดกรองตำหนิ 8 ชนิดให้ "
            "(รอยแตกลายงา, สิ่งแปลกปลอม, ผิวลอก, ผิวเป็นหลุม, สะเก็ดรีด, รอยขีดข่วน, สนิม, รอยแตกร้าว)</div>"
            "<div class='hd-note'>prototype เพื่อการศึกษา · ตรวจ “สนิม” ได้ดีที่สุด ชนิดอื่นบนภาพถ่ายจริงยังพลาดได้บ่อย · "
            "ผลเป็นเพียงตัวช่วยคัดกรอง ห้ามใช้ตัดสินคุณภาพชิ้นงานจริง</div>"
            "</div>"
        )

        cur_in = gr.State(None)     # ภาพล่าสุดที่ตรวจ (ไว้ให้ feedback)
        # พารามิเตอร์ทั้งหมดตั้งค่าที่ "ดีที่สุด" ไว้แล้ว — ผู้ใช้ไม่ต้องเลือกเอง
        #   โมเดล = train-real3 (ตัวแรกใน _STAGE2_MODELS ที่ weights มีจริง)
        #   ความไว = มาตรฐาน (thresholds_real3.json จูนเน้น recall อยู่แล้ว) · ไม่เปิด TTA/multiscale (ช้า)
        model_sel = gr.State(model_choices[0] if model_choices else None)
        sens = gr.State("มาตรฐาน")
        conf = gr.State(0.4)
        detailed = gr.State(False)
        gate_on = gr.State(steel_gate.available())
        multiscale = gr.State(False)

        # ===== รับภาพ: อัปโหลด / ถ่ายภาพ =====
        with gr.Tabs():
            with gr.Tab("อัปโหลดภาพ"):
                inp = gr.Image(type="numpy", label="ภาพเหล็กที่จะตรวจ", height=300,
                               sources=["upload", "clipboard"])
                gr.HTML("<div class='hint'>ลากไฟล์มาวาง · กดเลือกไฟล์ · หรือวาง (Ctrl+V) "
                        "— ระบบตรวจให้อัตโนมัติ</div>")
                btn = gr.Button("ตรวจสอบ", variant="primary", size="lg")
                if demo_samples or lab_samples or real_samples:
                    gr.Examples(examples=(demo_samples + lab_samples + real_samples), inputs=inp,
                                label="ภาพตัวอย่าง (กดเพื่อตรวจ)", examples_per_page=16)

            with gr.Tab("ถ่ายภาพ"):
                cam = gr.Image(type="numpy", label="กล้อง", height=340, sources=["webcam"])
                gr.HTML("<div class='hint'>อนุญาตให้เบราว์เซอร์ใช้กล้อง → เล็งไปที่ผิวเหล็ก → "
                        "<b>กดปุ่มถ่าย (วงกลม) ที่มุมล่างของภาพกล้อง</b> → ระบบตรวจให้อัตโนมัติ</div>")

        # ===== ผลตรวจ (โหมด อัปโหลด + ถ่ายภาพ) =====
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=300):
                status = gr.HTML(_empty_banner())
            with gr.Column(scale=5, min_width=300):
                table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["str"] * 5,
                                     row_count=(1, "dynamic"),
                                     interactive=False, wrap=True,
                                     elem_classes=["res-table"],
                                     label="รายการตำหนิ (เรียงตามความเสี่ยง)")

        # ----- สาเหตุที่พบบ่อย + คำแนะนำ ต่อชนิดตำหนิที่เจอ (ว่างเมื่อไม่พบตำหนิ) -----
        causes_box = gr.HTML()

        # ----- ภาพผลลัพธ์ (เต็มความกว้าง ปรับตามอัตราส่วนภาพ) -----
        out_img = gr.Image(type="numpy", label="ผลลัพธ์",
                           interactive=False, elem_classes=["result-img"])
        gr.HTML("<div class='legend'>"
                "<span class='lg lg-def'>กรอบแดง = ตำหนิที่ยืนยัน</span>"
                "<span class='lg lg-may'>กรอบส้ม = อาจมี (ความมั่นใจต่ำ)</span>"
                "</div>")

        # ----- ดาวน์โหลดผล + feedback -----
        res_state = gr.State({})
        with gr.Row():
            dl = gr.DownloadButton("ดาวน์โหลดผล (ภาพ + JSON)", visible=False, size="sm")
        with gr.Group(elem_classes=["fb"]):
            gr.HTML("<div class='fb-h'>ผลตรวจนี้เป็นอย่างไร? (ช่วยพัฒนาระบบ)</div>")
            with gr.Row():
                fb_rate = gr.Radio(["ถูกต้อง", "ผิด / ไม่ครบ", "ภาพนี้ไม่ควรตรวจ"],
                                   label=None, show_label=False, scale=3)
                fb_send = gr.Button("ส่ง feedback", size="sm", scale=1)
            fb_comment = gr.Textbox(label=None, show_label=False, lines=1,
                                    placeholder="ความเห็นเพิ่มเติม (ถ้ามี) เช่น ตำหนิที่ระบบพลาด")
            fb_msg = gr.Markdown()

        gr.HTML("<div class='foot'>2 ขั้นตอน: DMS46 หาพื้นที่โลหะ → YOLO11n ตรวจตำหนิ 8 ชนิด · "
                "โมเดลเทรนจาก NEU-DET + Roboflow (+ ภาพถ่ายจริงสำหรับตัว “ปรับโดเมน”) · "
                "feedback/ภาพที่ส่ง เก็บในเครื่องนี้ (demo_logs/)</div>")

        outputs = [out_img, status, table, causes_box, res_state]

        def _wire(trigger, img_comp):
            ins = [img_comp, conf, detailed, sens, model_sel, gate_on, multiscale]
            (trigger(lambda x: x, img_comp, cur_in)
             .then(analyze, inputs=ins, outputs=outputs, show_progress="minimal")
             .then(prepare_download, inputs=[out_img, res_state], outputs=dl))

        _wire(btn.click, inp)
        _wire(inp.change, inp)
        _wire(cam.change, cam)

        fb_send.click(submit_feedback,
                      inputs=[cur_in, out_img, res_state, fb_rate, fb_comment], outputs=fb_msg)
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
        print("  หมายเหตุกล้อง: การถ่ายจากกล้องบนมือถือ/เครื่องอื่นผ่าน http://<LAN-IP> จะถูกเบราว์เซอร์บล็อก")
        print("               ถ้าจะให้ถ่ายกล้องได้ ใช้  python app.py --share  (ลิงก์ *.gradio.live เป็น https)")
    if args.share:
        print("กำลังสร้างลิงก์สาธารณะ *.gradio.live ... (รอสักครู่ — ลิงก์นี้ถ่ายจากกล้องได้)")
    print()

    build_ui().queue().launch(
        server_name=host, server_port=args.port, share=args.share,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray"), css=_CSS)
