"""
mobile_api.py — REST API ให้แอปมือถือ (Flutter) เรียกตรวจตำหนิ โดยไม่ต้องรัน Gradio

ต้นแบบแอปมือถือ: D:\\mob_app\\steel_defect_scan (Flutter, เรียก API นี้ผ่าน http)
ใช้โมเดล/threshold ชุดเดียวกับหน้าเว็บ (app.py) — ตัวแรกใน _STAGE2_MODELS ที่มี weights จริง
(ปัจจุบันคือ train-real3 — ดู README หัวข้อ "2 ราง โมเดล Stage 2")

รัน:
    python mobile_api.py                  # http://0.0.0.0:8000
    python mobile_api.py --port 8001

ทดสอบเร็ว ๆ (PowerShell):
    curl.exe -F "file=@demo_samples/rust/rust_1_conf94.jpg" http://127.0.0.1:8000/detect

Endpoints:
    GET  /health        -> {"ok": true, "model": "..."}
    POST /detect         (multipart "file") -> pipeline เต็ม (Stage 1 หาพื้นที่เหล็ก + Stage 2)
                          ใช้กับภาพถ่ายเดี่ยว (แท็บ Gallery/Camera ของแอป)
    POST /detect_fast    (multipart "file") -> Stage 2 อย่างเดียว (ข้าม Stage 1) เร็วกว่ามาก
                          ใช้กับเฟรมกล้องสด (แท็บ Live ของแอป)

Response ของทั้งสอง endpoint (ตรวจ):
    {
      "model": "ปรับโดเมน v3 — ภาพถ่ายจริง (แนะนำ)",
      "detections": [
        {"class": "rust", "name_th": "สนิม", "risk": "สูง", "confidence": 0.94,
         "bbox": [x1, y1, x2, y2], "causes": "...", "advice": "..."},
        ...
      ]
    }
    /detect เพิ่ม "fallback_full_image" (bool) และ "metal_ratio" (float) จาก Stage 1
"""
import argparse

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import app as APP          # ใช้ตัวโหลดโมเดล + threshold ชุดเดียวกับหน้าเว็บ (ไม่ launch Gradio แค่ import)
import pipeline as P

api = FastAPI(title="Steel Defect Detection API",
              description="Backend สำหรับแอปมือถือ steel_defect_scan (Flutter)")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _read_upload(raw: bytes):
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "อ่านไฟล์ภาพไม่ได้ (ต้องเป็น jpg/png ที่ถูกต้อง)")
    return bgr


def _dets_to_json(dets, bbox_key):
    out = []
    for d in dets:
        di = P.DEFECT_INFO[d["class"]]
        out.append({
            "class": d["class"], "name_th": di["name_th"], "risk": di["risk"],
            "confidence": round(float(d["confidence"]), 4),
            "bbox": [round(v) for v in d[bbox_key]],
            "causes": di.get("causes", ""), "advice": di.get("advice", ""),
        })
    out.sort(key=lambda r: -r["confidence"])
    return out


@api.get("/health")
def health():
    mk = next(iter(APP._available_models()), None)
    return {"ok": mk is not None, "model": mk}


@api.post("/detect")
async def detect(file: UploadFile = File(...)):
    """pipeline เต็ม: Stage 1 (หาพื้นที่เหล็ก) + Stage 2 (ตรวจตำหนิ) — ภาพถ่ายเดี่ยว"""
    bgr = _read_upload(await file.read())
    mk = next(iter(APP._available_models()), None)
    if mk is None:
        raise HTTPException(503, "ไม่พบโมเดล Stage 2 (ดู runs/detect/ ในเครื่องเซิร์ฟเวอร์)")
    s1, s2, device = APP._ensure_models(mk)
    cc = APP._STATE["class_conf"] or None

    mask = P.run_stage1(s1, bgr, device)
    boxes, meta = P.build_regions(mask, bgr.shape)
    flat = []
    for (x, y, w, h) in boxes:
        crop = bgr[y:y + h, x:x + w]
        dets = P.run_stage2(s2, crop, APP._RAW_FLOOR, device, augment=False, class_conf=cc)
        for d in dets:
            cx1, cy1, cx2, cy2 = d["bbox_xyxy_crop"]
            d["bbox_xyxy_global"] = [round(cx1 + x, 1), round(cy1 + y, 1),
                                      round(cx2 + x, 1), round(cy2 + y, 1)]
        flat.extend(dets)
    kept = P.cross_region_nms(flat, iou_thresh=0.5)

    return {
        "model": mk,
        "fallback_full_image": meta["fallback_full_image"],
        "metal_ratio": round(meta["metal_ratio"], 4),
        "detections": _dets_to_json(kept, "bbox_xyxy_global"),
    }


@api.post("/detect_fast")
async def detect_fast(file: UploadFile = File(...)):
    """Stage 2 อย่างเดียว (ข้าม Stage 1) — เร็ว เหมาะกับเฟรมกล้องสด (แท็บ Live)"""
    bgr = _read_upload(await file.read())
    mk = next(iter(APP._available_models()), None)
    if mk is None:
        raise HTTPException(503, "ไม่พบโมเดล Stage 2 (ดู runs/detect/ ในเครื่องเซิร์ฟเวอร์)")
    _, s2, device = APP._ensure_models(mk)
    cc = APP._STATE["class_conf"] or None
    dets = P.run_stage2(s2, bgr, APP._RAW_FLOOR, device, augment=False, class_conf=cc)
    return {"model": mk, "detections": _dets_to_json(dets, "bbox_xyxy_crop")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
    except OSError:
        lan_ip = "<LAN-IP>"
    finally:
        s.close()

    print(f"Steel Defect API — เครื่องนี้: http://127.0.0.1:{args.port}")
    print(f"                    มือถือในวง LAN เดียวกัน: http://{lan_ip}:{args.port}")
    print(f"                    ตั้งใน steel_defect_scan (Flutter) เป็น base URL นี้")

    import uvicorn
    uvicorn.run(api, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
