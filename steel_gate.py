"""
steel_gate.py — Stage 0: ภาพนี้เป็น "พื้นผิวเหล็ก" ที่ควรตรวจตำหนิต่อไหม?

ใช้ classifier จาก train_gate.py (runs/classify/steel-gate/weights/best.pt)
ถ้ายังไม่ได้เทรน -> steel_prob() คืน None และ pipeline/เดโมทำงานเหมือนเดิม (ไม่มี gate)

    from steel_gate import steel_prob
    p = steel_prob(image_bgr)      # 0..1 = P(เป็นพื้นผิวเหล็ก) หรือ None ถ้าไม่มีโมเดล

เล่มจบไม่ใช้ไฟล์นี้ — เป็นส่วนเดโม/โปรดักต์
"""
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
GATE_WEIGHTS = BASE / "runs" / "classify" / "steel-gate" / "weights" / "best.pt"

# เกณฑ์ที่เดโมใช้ตีความ (ปรับได้)
LO = 0.35   # ต่ำกว่านี้ = "ไม่ใช่พื้นผิวเหล็ก" — ไม่ตรวจต่อ
HI = 0.60   # สูงกว่านี้ = เหล็กแน่ ๆ ; ระหว่าง LO..HI = ก้ำกึ่ง (ตรวจต่อ แต่ขึ้นหมายเหตุ)

_model = None
_loaded = False
_steel_idx = None


def available() -> bool:
    return GATE_WEIGHTS.exists()


def _load():
    global _model, _loaded, _steel_idx
    _loaded = True
    if not GATE_WEIGHTS.exists():
        return
    from ultralytics import YOLO
    _model = YOLO(str(GATE_WEIGHTS))
    names = _model.names          # {idx: name}
    for i, n in names.items():
        if str(n).lower() == "steel":
            _steel_idx = int(i)
    if _steel_idx is None:        # เผื่อชื่อคลาสไม่ตรง — เดาจากลำดับ (not_steel, steel)
        _steel_idx = 1 if len(names) > 1 else 0


def steel_prob(image_bgr) -> float | None:
    """คืน P(steel) 0..1 — หรือ None ถ้ายังไม่มีโมเดล gate"""
    if not _loaded:
        _load()
    if _model is None:
        return None
    rgb = image_bgr[:, :, ::-1]
    r = _model.predict(rgb, verbose=False)[0]
    probs = r.probs.data.detach().cpu().numpy().astype(float)
    return float(probs[_steel_idx])


def verdict(p: float | None) -> str:
    """'steel' | 'maybe' | 'not_steel' | 'unknown'"""
    if p is None:
        return "unknown"
    if p < LO:
        return "not_steel"
    if p < HI:
        return "maybe"
    return "steel"
