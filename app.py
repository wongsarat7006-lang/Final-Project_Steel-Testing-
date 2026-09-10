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


_C_OK, _C_MAYBE = (68, 68, 239), (11, 158, 245)   # BGR ~ #ef4444 / #f59e0b


_ALL_TH = [P.DEFECT_INFO[c]["name_th"] for c in P.DEFECT_CLASSES]   # 8 ชื่อไทย ตามลำดับคลาส


def _rt_card(n, found, err=None):
    """แผงผลสดโหมดเรียลไทม์ — โชว์ทั้ง 8 ชนิด + % ความมั่นใจสูงสุดในเฟรม (ชนิดที่มั่นใจสุด = ตัวหนา)
    n < 0 = error, 0 = ยังไม่พบ, >0 = จำนวนกล่องตำหนิในเฟรม"""
    if err:
        return f"<div class='rt-panel rt-err'>ประมวลผลเฟรมไม่สำเร็จ: {err}</div>"
    head = ("● ยังไม่พบตำหนิในเฟรม" if not n
            else f"● พบตำหนิ <b>{n}</b> จุด · {len(found)} ชนิด")
    top = max(found, key=found.get) if found else None
    rows = []
    for th in _ALL_TH:
        v = found.get(th, 0.0)
        rows.append(
            f"<div class='rt-row{' rt-row-on' if th == top else ''}'>"
            f"<span class='rt-name'>{th}</span>"
            f"<span class='rt-bar'><i style='width:{int(round(v * 100))}%'></i></span>"
            f"<span class='rt-pct'>{f'{v:.0%}' if v else '—'}</span></div>")
    return (f"<div class='rt-panel {'rt-hit' if n else 'rt-ok'}'>"
            f"<div class='rt-head'>{head}</div>{''.join(rows)}</div>")


def analyze_stream(frame_rgb):
    """โหมดเรียลไทม์: รัน Stage 2 บนเฟรมกล้องต่อเนื่อง (ข้าม Stage 1) — เร็วพอสำหรับดูสด
    ใช้ค่าคงที่เดียวกับหน้าอัปโหลด (โมเดลตัวแรกที่มี = train-real3, threshold รายคลาส)
    คืน (เฟรมพร้อมกรอบ+ป้ายไทย, การ์ดสรุปสด)"""
    if frame_rgb is None:
        return None, _rt_card(0, {})
    try:
        bgr = _to_bgr(frame_rgb)
        mk = next(iter(_available_models()), None)
        _, s2, device = _ensure_models(mk)
        cc = _STATE["class_conf"] or {}
        dets = P.run_stage2(s2, bgr, _RAW_FLOOR, device, augment=False, class_conf=None)
        S = max(bgr.shape[:2])
        lw = max(2, round(S / 380))
        fpx = int(min(48, max(16, S / 40)))
        found, n = {}, 0
        for d in dets:
            if d["confidence"] < max(_RAW_FLOOR, cc.get(d["class"], 0.4)):
                continue
            n += 1
            x1, y1, x2, y2 = (int(v) for v in d["bbox_xyxy_crop"])
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 255, 255), lw + 2)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), _C_OK, lw)
            di = P.DEFECT_INFO[d["class"]]
            ly = y1 - fpx - 6 if y1 - fpx - 6 >= 2 else y1 + 4
            bgr = P.draw_thai_text(bgr, f"{di['name_th']} {d['confidence']:.0%}",
                                   (max(x1, 3), ly), color_bgr=_C_OK, font_size=fpx)
            k = di["name_th"]
            found[k] = max(found.get(k, 0), d["confidence"])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), _rt_card(n, found)
    except Exception as e:
        return frame_rgb, _rt_card(-1, {}, err=str(e))


def analyze_screen(data_url):
    """โหมดเรียลไทม์ (ส่องหน้าจอ): รับเฟรม crop (data:URL) จากในกรอบ
    คืน (ภาพในกรอบ + กรอบตำหนิที่วาดแล้ว, การ์ดสรุป 8 ชนิด)"""
    if not data_url or "," not in data_url:
        return None, _rt_card(0, {})
    try:
        import base64
        raw = base64.b64decode(data_url.split(",", 1)[1])
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, _rt_card(0, {})
        return analyze_stream(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        return None, _rt_card(-1, {}, err=str(e))


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


# โหลดตอนเปิดหน้า — ตั้งโทนสี + โหมด "ส่องหน้าจอ" เต็มจอ + กรอบเล็งกลางจอ (แนว app มือถือ)
_JS_ONLOAD = (r"""
() => {
  const $ = id => document.getElementById(id);
  const ALL = __ALL_TH__;   // 8 ชื่อไทยตามลำดับคลาส
  try {
    const s = localStorage.getItem('steeldemo_theme');
    const dark = s ? s === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.body.classList.toggle('dark', dark);
    document.querySelectorAll('gradio-app, .gradio-container')
            .forEach(e => e.classList.toggle('dark', dark));
  } catch (e) {}

  const S = window.__ss = window.__ss || {stream:null, video:null, canvas:null, timer:null, busy:false};

  // ปุ่ม start/stop/close ผูกแบบ delegation — ทำงานแม้แท็บเพิ่งเรนเดอร์
  if (!window.__ssClickBound) {
    window.__ssClickBound = true;
    document.addEventListener('click', (e) => {
      const t = e.target;
      if (t.closest && t.closest('#ss_start')) {
        e.preventDefault(); console.log('[steeldemo] ส่องหน้าจอ: คลิก'); window.__startScreen();
      } else if (t.closest && t.closest('#ss_stop')) {
        e.preventDefault(); window.__stopScreen();
      } else if (t.closest && t.closest('#ss_expand')) {
        e.preventDefault();
        const st = $('rt_stage'); if (!st) return;
        const big = st.classList.toggle('big');
        try {
          if (big && document.documentElement.requestFullscreen) document.documentElement.requestFullscreen();
          else if (!big && document.fullscreenElement) document.exitFullscreen();
        } catch (err) {}
        setTimeout(() => { if (window.__resetScope) window.__resetScope(); }, 260);
      }
    }, true);
  }

  // กรอบเล็ง = สี่เหลี่ยม "ลากย้าย + ปรับขนาด" ได้ ทับได้ทั้งพื้นที่หน้าจอที่แชร์ (ตำแหน่งเก็บใน dataset px)
  function applyScope() {
    const sc = $('rt_scope'); if (!sc) return;
    sc.style.left = (parseFloat(sc.dataset.x) || 0) + 'px';
    sc.style.top = (parseFloat(sc.dataset.y) || 0) + 'px';
    sc.style.width = (parseFloat(sc.dataset.w) || 120) + 'px';
    sc.style.height = (parseFloat(sc.dataset.h) || 120) + 'px';
  }
  function clampScope() {
    const st = $('rt_stage'), sc = $('rt_scope');
    if (!st || !sc || !st.clientWidth) return;
    const W = st.clientWidth, H = st.clientHeight;
    let w = Math.max(60, Math.min(parseFloat(sc.dataset.w) || W * 0.5, W));
    let h = Math.max(60, Math.min(parseFloat(sc.dataset.h) || H * 0.5, H));
    let x = Math.max(0, Math.min(parseFloat(sc.dataset.x) || 0, W - w));
    let y = Math.max(0, Math.min(parseFloat(sc.dataset.y) || 0, H - h));
    sc.dataset.w = w; sc.dataset.h = h; sc.dataset.x = x; sc.dataset.y = y;
    applyScope();
  }
  function resetScope() {
    const st = $('rt_stage'), sc = $('rt_scope');
    if (!st || !sc || !st.clientWidth) return;
    const s = Math.round(Math.min(st.clientWidth, st.clientHeight) * 0.38);
    sc.dataset.w = s; sc.dataset.h = s;
    sc.dataset.x = Math.round((st.clientWidth - s) / 2);
    sc.dataset.y = Math.round((st.clientHeight - s) / 2);
    applyScope();
  }
  window.__resetScope = resetScope;
  function initScopeDrag() {
    const sc = $('rt_scope'); if (!sc || sc.__wired) return; sc.__wired = true;
    let mode = null, px = 0, py = 0, ox = 0, oy = 0, ow = 0, oh = 0;
    const pt = e => e.touches ? e.touches[0] : e;
    const down = (e, m) => {
      mode = m; const p = pt(e); px = p.clientX; py = p.clientY;
      ox = parseFloat(sc.dataset.x) || 0; oy = parseFloat(sc.dataset.y) || 0;
      ow = parseFloat(sc.dataset.w) || 0; oh = parseFloat(sc.dataset.h) || 0;
      e.preventDefault(); e.stopPropagation();
    };
    const mv = e => {
      if (!mode) return;
      const p = pt(e), dx = p.clientX - px, dy = p.clientY - py;
      if (mode === 'move') { sc.dataset.x = ox + dx; sc.dataset.y = oy + dy; }
      else { sc.dataset.w = ow + dx; sc.dataset.h = oh + dy; }
      clampScope(); e.preventDefault();
    };
    const up = () => { mode = null; };
    sc.addEventListener('mousedown', e => { if (e.target === sc) down(e, 'move'); });
    sc.addEventListener('touchstart', e => { if (e.target === sc) down(e, 'move'); }, {passive: false});
    const h = sc.querySelector('.rt-scope-handle');
    if (h) {
      h.addEventListener('mousedown', e => down(e, 'resize'));
      h.addEventListener('touchstart', e => down(e, 'resize'), {passive: false});
    }
    window.addEventListener('mousemove', mv);
    window.addEventListener('touchmove', mv, {passive: false});
    window.addEventListener('mouseup', up);
    window.addEventListener('touchend', up);
  }
  initScopeDrag();

  // คืนพิกัดกรอบเทียบ "วิดีโอ" (พิกัดจริงของภาพ) และ "เวที" (พิกัดบนจอ)
  function rectVsVideo() {
    const sc = $('rt_scope'), v = S.video; if (!sc || !v || !v.videoWidth) return null;
    const sr = sc.getBoundingClientRect(), vr = v.getBoundingClientRect();
    if (!vr.width) return null;
    const kx = v.videoWidth / vr.width, ky = v.videoHeight / vr.height;
    let x = (sr.left - vr.left) * kx, y = (sr.top - vr.top) * ky;
    let w = sr.width * kx, h = sr.height * ky;
    x = Math.max(0, Math.min(x, v.videoWidth - 1)); y = Math.max(0, Math.min(y, v.videoHeight - 1));
    w = Math.max(1, Math.min(w, v.videoWidth - x)); h = Math.max(1, Math.min(h, v.videoHeight - y));
    return {x, y, w, h};
  }
  function rectVsStage() {
    const sc = $('rt_scope'), st = $('rt_stage'); if (!sc || !st) return null;
    const sr = sc.getBoundingClientRect(), gr = st.getBoundingClientRect();
    return {x: sr.left - gr.left, y: sr.top - gr.top, w: sr.width, h: sr.height};
  }

  window.__ssPush = () => {
    if (!S.video || S.busy) return;
    const box = document.querySelector('#ss_frame textarea'); if (!box) return;
    const r = rectVsVideo(); if (!r) return;
    const outW = Math.min(768, Math.max(1, Math.round(r.w))), k = outW / r.w;
    S.canvas.width = outW; S.canvas.height = Math.max(1, Math.round(r.h * k));
    S.canvas.getContext('2d').drawImage(S.video, r.x, r.y, r.w, r.h, 0, 0, S.canvas.width, S.canvas.height);
    S.busy = true;
    box.value = S.canvas.toDataURL('image/jpeg', 0.72);
    box.dispatchEvent(new Event('input', {bubbles: true}));
  };

  window.__startScreen = async () => {
    const warn = $('rt_warn'), stage = $('rt_stage'), hint = $('rt_hint');
    const fail = (msg) => {
      if (warn) { warn.style.display = 'block'; warn.classList.add('rt-warn-hot'); }
      if (hint) hint.textContent = msg || 'เปิดการส่องหน้าจอไม่สำเร็จ';
    };
    console.log('[steeldemo] __startScreen · secure=' + window.isSecureContext +
                ' · api=' + !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia));
    if (hint) hint.textContent = 'กำลังขอสิทธิ์แชร์หน้าจอ…';
    if (stage) stage.style.display = 'block';
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      fail('เบราว์เซอร์นี้ไม่รองรับการส่องหน้าจอ (ต้อง https/localhost + เบราว์เซอร์บนคอมพิวเตอร์)');
      return;
    }
    try {
      // ขอ "ทั้งหน้าจอ" เป็นค่าเริ่มต้น — ผู้ใช้กดยืนยันครั้งเดียว ไม่ต้องไล่เลือกแท็บ
      const s = await navigator.mediaDevices.getDisplayMedia(
        {video: {frameRate: 8, displaySurface: 'monitor'}, audio: false, selfBrowserSurface: 'exclude'});
      const v = document.createElement('video');
      v.autoplay = true; v.muted = true; v.playsInline = true; v.srcObject = s;
      await v.play();
      const holder = $('rt_video_holder');
      if (holder) { holder.innerHTML = ''; holder.appendChild(v); }
      S.stream = s; S.video = v; S.canvas = document.createElement('canvas'); S.busy = false;
      s.getVideoTracks()[0].addEventListener('ended', () => window.__stopScreen());
      initScopeDrag();
      setTimeout(resetScope, 120);
      v.addEventListener('loadedmetadata', () => setTimeout(resetScope, 60));
      if (S.timer) clearInterval(S.timer);
      S.timer = setInterval(window.__ssPush, 900);
      if (hint) hint.textContent = 'ลากกรอบไปครอบภาพเหล็กในอีกแท็บ · มุมล่างขวา = ปรับขนาด';
      if (warn) { warn.classList.remove('rt-warn-hot'); warn.style.display = 'none'; }
    } catch (e) {
      console.warn('[steeldemo] getDisplayMedia error', e);
      fail((e && e.name === 'NotAllowedError')
           ? 'คุณยกเลิก หรือเบราว์เซอร์ปฏิเสธการแชร์หน้าจอ'
           : 'ส่องหน้าจอไม่ได้ (' + (e && e.name || 'error') + ') — ต้อง https/localhost + เบราว์เซอร์บนคอมพิวเตอร์');
    }
  };
  window.__stopScreen = () => {
    if (S.timer) clearInterval(S.timer); S.timer = null;
    if (S.stream) S.stream.getTracks().forEach(t => t.stop());
    S.stream = null; S.video = null; S.canvas = null; S.busy = false;
    const stage = $('rt_stage');
    if (stage) { stage.style.display = 'none'; stage.classList.remove('big'); }
    try { document.fullscreenElement && document.exitFullscreen(); } catch (e) {}
    const holder = $('rt_video_holder'); if (holder) holder.innerHTML = '';
  };

  try { if (!window.isSecureContext) { const w = $('rt_warn'); if (w) w.style.display = 'block'; } } catch (e) {}
}
""").replace("__ALL_TH__", json.dumps(_ALL_TH, ensure_ascii=False))
# ปุ่มสลับโทนสว่าง/มืด
_JS_TOGGLE = """
() => {
  const on = !document.body.classList.contains('dark');
  document.body.classList.toggle('dark', on);
  document.querySelectorAll('gradio-app, .gradio-container')
          .forEach(e => e.classList.toggle('dark', on));
  try { localStorage.setItem('steeldemo_theme', on ? 'dark' : 'light'); } catch (e) {}
}
"""
# หลัง Python ประมวลผลเฟรมส่องหน้าจอเสร็จ -> ปลดล็อกให้ส่งเฟรมถัดไป (backpressure)
_JS_SS_DONE = "() => { if (window.__ss) window.__ss.busy = false; }"

_CSS = """
:root{
  color-scheme:light;
  --bg:#ffffff; --surface:#fbfcfd; --card:#ffffff;
  --border:#e3e8ee; --border-soft:#eef2f6;
  --fg:#0f172a; --fg-2:#334155; --fg-3:#475569; --fg-faint:#94a3b8;
  --shadow:0 1px 3px rgba(15,23,42,.06);
  --tab-fg:#64748b; --tab-fg-on:#0f172a; --tab-bg-on:#eef4ff; --tab-bar:#2563eb;
  --limits-bg:#fff8ef; --limits-bd:#fde3c0; --limits-fg:#7a5320; --limits-fg-b:#7a3f0e;
}
body.dark{
  color-scheme:dark;
  --bg:#0e1420; --surface:#161d2b; --card:#1a2232;
  --border:#2b3648; --border-soft:#232d3d;
  --fg:#f1f5f9; --fg-2:#cbd5e1; --fg-3:#aab6c6; --fg-faint:#8090a4;
  --shadow:0 1px 3px rgba(0,0,0,.4);
  --tab-fg:#93a2b8; --tab-fg-on:#f1f5f9; --tab-bg-on:#1f2b3d; --tab-bar:#60a5fa;
  --limits-bg:#2a2213; --limits-bd:#4a3a1e; --limits-fg:#e7c58c; --limits-fg-b:#f1d5a4;
}
footer{display:none!important}
body,.gradio-container{background:var(--bg)!important;color:var(--fg)}
.gradio-container{max-width:1100px!important;margin:0 auto!important;padding:12px 14px 32px!important}

/* ===== แท็บหัวข้อ — ให้เด่นชัด ===== */
.tabs>.tab-nav, .tab-nav{
  border-bottom:2px solid var(--border)!important; gap:6px!important; margin-bottom:16px!important}
.tab-nav button{
  font-size:16.5px!important; font-weight:700!important; color:var(--tab-fg)!important;
  padding:11px 22px!important; border:none!important; background:transparent!important;
  border-radius:9px 9px 0 0!important; opacity:1!important}
.tab-nav button:hover{color:var(--tab-fg-on)!important; background:var(--surface)!important}
.tab-nav button.selected{
  color:var(--tab-fg-on)!important; background:var(--tab-bg-on)!important;
  box-shadow:inset 0 -3px 0 var(--tab-bar)!important}

/* ===== ปุ่มสลับโทน (มุมขวาบน) ===== */
.topbar{align-items:flex-start!important; gap:8px!important}
.themebtn{flex:none!important; min-width:0!important}
.themebtn button{font-size:13px!important; padding:7px 12px!important}

/* ภาพผลตรวจ */
.result-img{border:1px solid var(--border);border-radius:12px;background:var(--surface);min-height:220px}
.result-img img{object-fit:contain!important;max-height:72vh!important}
/* หัวเรื่อง */
.hd{padding:6px 2px 12px}
.hd-title{font-size:24px;font-weight:800;color:var(--fg);letter-spacing:.2px;line-height:1.25}
.hd-sub{font-size:14.5px;color:var(--fg-3);margin-top:6px;line-height:1.55}
.hd-note{font-size:12.5px;color:var(--fg-faint);margin-top:5px}
.hint{font-size:13px;color:var(--fg-3);margin:-2px 0 10px;line-height:1.55}
/* การ์ดสรุปผล */
.rc{border:1px solid var(--border);border-left:6px solid var(--border);border-radius:12px;
    padding:16px 18px;background:var(--card);box-shadow:var(--shadow)}
.rc-kicker{font-size:11px;letter-spacing:.12em;color:var(--fg-faint);text-transform:uppercase;font-weight:700}
.rc-title{font-size:19px;font-weight:800;color:var(--fg);line-height:1.35;margin-top:4px}
.rc-sub{font-size:14.5px;color:var(--fg-2);margin-top:8px;line-height:1.6}
/* legend ใต้ภาพผล */
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:10px 2px 2px}
.lg{font-size:13px;color:var(--fg-3);display:flex;align-items:center}
.lg::before{content:"";width:13px;height:13px;border-radius:3px;margin-right:7px;flex:none}
.lg-def::before{background:#ef4444}
.lg-may::before{background:#f59e0b}
.foot{font-size:12.5px;color:var(--fg-faint);line-height:1.65;padding:16px 2px 2px;
    border-top:1px solid var(--border-soft);margin-top:20px}
/* กล่องขอบเขต/ข้อจำกัด */
.limits{border:1px solid var(--limits-bd);background:var(--limits-bg);border-radius:10px;
    padding:12px 16px;margin:2px 2px 14px;font-size:13.5px;color:var(--limits-fg);line-height:1.6}
.limits b{color:var(--limits-fg-b)}
/* กล่องสาเหตุ/คำแนะนำต่อชนิดตำหนิ */
.causes{border:1px solid var(--border);background:var(--surface);border-radius:12px;
    padding:14px 18px;margin:12px 2px 2px;font-size:14px;color:var(--fg-2);line-height:1.65}
.causes-h{font-size:14.5px;font-weight:800;color:var(--fg);margin-bottom:11px}
.causes-h span{display:block;font-size:12px;font-weight:400;color:var(--fg-faint);margin-top:2px}
.causes-item{padding:11px 0;border-top:1px solid var(--border)}
.causes-item:first-of-type{border-top:0;padding-top:2px}
.causes-item b{color:var(--fg);font-size:15px}
.causes-risk{font-size:12.5px;color:var(--fg-3);margin-left:8px}
.causes-adv{color:var(--fg-3);margin-top:3px}
/* เรียลไทม์ */
#ss_frame{display:none!important}
.rt-warn{border:1px solid #fcd9b0;background:#fff7ec;color:#8a4b12;border-radius:10px;
    padding:11px 15px;margin:6px 2px 10px;font-size:13px;line-height:1.7}
.rt-warn code{background:rgba(0,0,0,.06);padding:1px 5px;border-radius:4px;font-size:12px}
.rt-warn.rt-warn-hot{border-color:#ef4444;background:#fef2f2;color:#b91c1c;font-weight:600}
/* พรีวิวหน้าจอที่แชร์ + กรอบเล็ง (ลากย้าย/ปรับขนาดได้) */
.rt-stage{position:relative;margin-top:10px;border-radius:12px;overflow:hidden;
    background:#000;border:1px solid var(--border);user-select:none}
#rt_video_holder video{width:100%;display:block;max-height:76vh;object-fit:contain;background:#000}
/* ปุ่ม "ขยายเต็มจอ" — พรีวิวคลุมทั้งหน้าต่าง เล็งกรอบได้ละเอียด */
.rt-stage.big{position:fixed;inset:0;z-index:2147483000;margin:0;border:0;border-radius:0}
.rt-stage.big #rt_video_holder video{width:100vw;height:100vh;max-height:none;object-fit:contain}
.rt-scope{position:absolute;left:20%;top:20%;width:56%;height:56%;box-sizing:border-box;z-index:2;
    cursor:move;touch-action:none;border:2px solid #22d3ee;box-shadow:0 0 0 9999px rgba(0,0,0,.42)}
.rt-scope-handle{position:absolute;right:-12px;bottom:-12px;width:24px;height:24px;
    background:#22d3ee;border:3px solid #fff;border-radius:6px;cursor:se-resize;touch-action:none}
.rt-stage-hint{position:absolute;left:8px;top:8px;z-index:3;font-size:11.5px;color:#fff;
    background:rgba(0,0,0,.6);padding:4px 9px;border-radius:7px;pointer-events:none}
/* แผงผลสด — ลิสต์ทุกชนิด + แถบ % */
.rt-panel{border:1px solid var(--border);border-left:5px solid var(--border);
    border-radius:12px;background:var(--card);padding:12px 14px;margin-top:8px}
.rt-panel.rt-ok{border-left-color:#22c55e}
.rt-panel.rt-hit{border-left-color:#ef4444}
.rt-panel.rt-err{border-left-color:#f59e0b;color:#b45309;font-size:13px;font-weight:600;padding:12px 16px}
.rt-head{font-size:15px;font-weight:800;color:var(--fg);margin-bottom:8px}
.rt-panel.rt-ok .rt-head{color:#16a34a}
.rt-panel.rt-hit .rt-head{color:#dc2626}
.rt-row{display:flex;align-items:center;gap:8px;padding:3px 0;font-size:13px;color:var(--fg-3)}
.rt-name{flex:0 0 128px;font-family:ui-monospace,Menlo,Consolas,monospace}
.rt-bar{flex:1;height:8px;border-radius:5px;background:var(--surface);overflow:hidden}
.rt-bar>i{display:block;height:100%;background:#cbd5e1;transition:width .2s}
.rt-pct{flex:0 0 42px;text-align:right;font-variant-numeric:tabular-nums}
.rt-row-on{color:var(--fg);font-weight:800}
.rt-row-on .rt-bar>i{background:#ef4444}
.rt-desc{border:1px solid var(--border);background:var(--surface);border-radius:12px;
    padding:13px 16px;margin-top:12px;font-size:13px;color:var(--fg-3);line-height:1.7}
.rt-desc b{color:var(--fg-2)}
/* feedback */
.fb{border:1px solid var(--border);border-radius:12px;padding:14px 18px;margin-top:12px;background:var(--card)}
.fb-h{font-size:14.5px;font-weight:800;color:var(--fg);margin-bottom:4px}
/* ตารางผล (gr.Dataframe) — เลื่อนแนวนอนได้เมื่อจอแคบ */
.res-table .table-wrap, .res-table table{font-size:14px!important}
.res-table{overflow-x:auto}
/* ===== จอมือถือ / จอแคบ ===== */
@media (max-width:640px){
  .gradio-container{padding:8px 10px 24px!important}
  .hd-title{font-size:20px}
  .hd-sub{font-size:13px}
  .tab-nav button{font-size:15px!important;padding:9px 15px!important}
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

    with gr.Blocks(title="ตรวจตำหนิพื้นผิวเหล็ก", js=_JS_ONLOAD) as demo:
        with gr.Row(elem_classes=["topbar"]):
            gr.HTML(
                "<div class='hd'>"
                "<div class='hd-title'>ตรวจจับตำหนิพื้นผิวเหล็ก</div>"
                "<div class='hd-sub'>อัปโหลดหรือถ่ายภาพผิวเหล็ก ระบบจะคัดกรองตำหนิ 8 ชนิดให้ "
                "(รอยแตกลายงา, สิ่งแปลกปลอม, ผิวลอก, ผิวเป็นหลุม, สะเก็ดรีด, รอยขีดข่วน, สนิม, รอยแตกร้าว)</div>"
                "<div class='hd-note'>prototype เพื่อการศึกษา · ตรวจ “สนิม” ได้ดีที่สุด ชนิดอื่นบนภาพถ่ายจริงยังพลาดได้บ่อย · "
                "ผลเป็นเพียงตัวช่วยคัดกรอง ห้ามใช้ตัดสินคุณภาพชิ้นงานจริง</div>"
                "</div>"
            )
            theme_btn = gr.Button("🌗 สลับโทนสว่าง/มืด", size="sm", scale=0,
                                  elem_classes=["themebtn"])
        theme_btn.click(fn=None, inputs=None, outputs=None, js=_JS_TOGGLE)

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

        # ===== รับภาพ: อัปโหลด / ถ่ายภาพ / เรียลไทม์ =====
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

            with gr.Tab("เรียลไทม์ (ส่องหน้าจอ)"):
                gr.HTML(
                    "<div class='hint'>สำหรับใช้ <b>2 หน้าต่าง/แท็บ</b> — หน้าต่างหนึ่งเปิดภาพเหล็ก · แท็บนี้แสดงผลตรวจสด<br>"
                    "กด <b>ส่องหน้าจอ</b> → เลือก “ทั้งหน้าจอ” → <b>พรีวิวทั้งจอ</b>ขึ้นในหน้านี้ (เห็นทุกหน้าต่างรวมถึงภาพเหล็ก) "
                    "→ <b>ลากกรอบสีฟ้า</b>ไปวางบนภาพเหล็กในพรีวิว (ลากตัวกรอบ=ย้าย · มุมล่างขวา=ปรับขนาด) · "
                    "กด <b>⛶ ขยาย/ย่อพรีวิว</b> ให้พรีวิวเต็มหน้าต่างเพื่อเล็งละเอียด · ตรวจในกรอบทุก ~0.9 วินาที</div>"
                    "<div id='rt_warn' class='rt-warn' style='display:none'>"
                    "⚠️ <b>ส่องหน้าจอไม่ได้</b> — เบราว์เซอร์ยอมให้ทำเฉพาะหน้าที่เป็น <b>https</b> หรือ <b>localhost</b><br>"
                    "• รัน <code>python app.py --share</code> แล้วเปิดลิงก์ <code>https://…gradio.live</code><br>"
                    "• หรือเปิด <code>http://127.0.0.1:7860</code> บนเครื่องนี้โดยตรง<br>"
                    "• หรือใน Chrome: <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code> "
                    "→ ใส่ <code>http://192.168.1.102:7860</code> → Enabled → รีสตาร์ตเบราว์เซอร์</div>")
                with gr.Row():
                    ss_start = gr.Button("🖥️ ส่องหน้าจอ", variant="primary", size="lg",
                                         elem_id="ss_start")
                    ss_expand = gr.Button("⛶ ขยาย/ย่อพรีวิว", size="lg", elem_id="ss_expand")
                    ss_stop = gr.Button("■ หยุด", size="lg", elem_id="ss_stop")
                ss_frame = gr.Textbox(elem_id="ss_frame")     # ซ่อน — รับ data:URL (crop) จาก JS
                gr.HTML(
                    "<div id='rt_stage' class='rt-stage' style='display:none'>"
                    "<div id='rt_video_holder'></div>"
                    "<div id='rt_hint' class='rt-stage-hint'>ลากกรอบไปครอบภาพเหล็กในอีกแท็บ · มุมล่างขวา = ปรับขนาด</div>"
                    "<div id='rt_scope' class='rt-scope'><div class='rt-scope-handle'></div></div>"
                    "</div>")
                rt_out = gr.Image(type="numpy", label="ภาพที่กำลังส่อง (ในกรอบ) + กรอบตำหนิ",
                                  interactive=False, elem_classes=["result-img"])
                rt_txt = gr.HTML(_rt_card(0, {}))
                gr.HTML(
                    "<div class='rt-desc'>"
                    "<b>ใช้ยังไง</b> — เปิด <b>2 แท็บ</b>: แท็บหนึ่งเปิดภาพเหล็กที่จะทดสอบ · แท็บนี้กด <b>ส่องหน้าจอ</b> "
                    "→ เลือก “ทั้งหน้าจอ” → พรีวิวหน้าจอจะขึ้นด้านบน พร้อม<b>กรอบสีฟ้าที่ลาก/ปรับขนาดได้</b> "
                    "→ เลื่อนกรอบไปครอบภาพเหล็กในอีกแท็บ<br>"
                    "<b>อ่านผลยังไง</b> — ช่อง “ภาพที่กำลังส่อง” = สิ่งที่อยู่ในกรอบ + กรอบแดงรอบตำหนิที่เจอ (อัปเดตทุก ~0.9 วิ) · "
                    "แผงด้านล่างไล่ทั้ง 8 ชนิด ตัวเลข % = ความมั่นใจสูงสุดของชนิดนั้น · "
                    "ชนิดที่มั่นใจสุด = ตัวหนา/แถบแดง · “—” = ไม่พบ<br>"
                    "<b>ข้อจำกัด</b> — โมเดลแม่นเรื่อง<b>สนิม</b>ที่สุด อีก 7 ชนิดบนภาพถ่ายจริงยังพลาดได้บ่อย · "
                    "ภาพย่อเล็ก/เบลอ ทำให้เตือนผิดหรือพลาดได้ · ให้ยึดผลจากแท็บ “อัปโหลดภาพ” เป็นหลัก<br>"
                    "<b>ต้องเปิดผ่าน https หรือ localhost</b> เท่านั้น การส่องหน้าจอถึงจะทำงาน (ดูกรอบเตือนด้านบนถ้าใช้ไม่ได้)"
                    "</div>")

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

        # โหมดเรียลไทม์ — ส่องหน้าจอ. ปุ่มผูกผ่าน click delegation ใน _JS_ONLOAD (กันแท็บ lazy-render)
        # ทาง .click(js=) ไว้เป็นทางสำรอง
        ss_start.click(fn=None, js="() => window.__startScreen && window.__startScreen()")
        ss_stop.click(fn=None, js="() => window.__stopScreen && window.__stopScreen()")
        ss_frame.change(analyze_screen, inputs=[ss_frame], outputs=[rt_out, rt_txt],
                        show_progress="hidden", concurrency_limit=1).then(fn=None, js=_JS_SS_DONE)
        demo.load(fn=None, js=_JS_ONLOAD)   # สำรอง เผื่อ Blocks(js=) ไม่ทำงานในบางเวอร์ชัน
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
