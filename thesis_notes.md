# บันทึกสำหรับเขียนเล่ม — กรอบคิด 3 ข้อ

สรุปการตัดสินใจเชิงกรอบของโปรเจค ไว้อ้างตอนเขียนบท "ขอบเขต / วิธีดำเนินการ / การทดลอง"
(ปรับปรุง 2026-09-02)

---

## 1. Use case จริง

**คัดกรองสภาพผิวเหล็กก่อนนำไปใช้ / รับเข้าคลัง — สเกลโกดัง / ร้านเหล็ก / โรงกลึง**

### Scenario
เหล็กเส้น / แผ่น / คอยล์ ที่ส่งเข้าโรงกลึงหรือโกดัง → พนักงานถ่ายรูปด้วยมือถือ →
ระบบคัดกรองว่ามีตำหนิชนิดใด **ก่อนรับเข้า / ก่อนตัดใช้งาน**
ขอบเขตที่เคลม = **prototype ผู้ช่วยคัดกรอง (screening assistant)** ไม่ใช่ระบบตรวจสอบติดตั้งใช้งานจริง

### ทำไม use case นี้แมตช์กับระบบที่ทำ
| เหตุผล | รายละเอียด |
|---|---|
| ครบทั้ง 8 คลาส | ตำหนิจากการผลิตที่ติดมากับเหล็ก (6 คลาส NEU) + สนิมจากขนส่ง/เก็บ (rust) + รอยแตกจากการยก (crack) |
| ภาพมีบริบทฉาก | ถ่ายมือถือใต้แสงโรงงาน → มีพื้น/ผนัง/พาเลทในเฟรม → **Stage 1 มีงานทำ** (กันพื้นหลังที่ไม่ใช่เหล็ก) |
| เป็นงานคัดกรอง | ไม่ต้องวัดขนาดตำหนิละเอียด → ความแม่นระดับ mAP50 0.76 พอใช้งานได้ |
| UI ตรงรูปแบบ | `app.py` (Gradio) = อัปรูป → ได้ผล ตรงกับ use case พอดี |

### ทำไม "โรงงานผลิตเหล็ก" ใหญ่เกิน + ไม่แมตช์
| โรงงานผลิตเหล็กต้องมี | โปรเจคนี้มีไหม |
|---|---|
| กล้องติดตั้งตายตัว + แสงคุม + sync สายพาน | ไม่มี |
| ทำงาน real-time + ต่อ PLC ดีดของเสีย | ไม่มี |
| ข้อมูลจากไลน์จริงไว้ validate | ไม่มี |
| **Stage 1 (หา region เหล็ก)** | **ไม่จำเป็นเลย** — ทุกเฟรมบนสายพาน = เหล็ก 100% |

→ ข้อสุดท้ายเป็นตัวตัดสิน: ถ้าเป็นสายพานผลิต Stage 1 ไม่มีเหตุผลจะมีอยู่ → สถาปัตยกรรม 2-stage พังทั้งอัน

---

## 2. Stage 1 (DMS46) — เก็บโค้ดไว้ ทำเป็น ablation study

**ไม่ตัดทิ้งล่วงหน้า** — ให้ผลจาก `real_test/` เป็นตัวตัดสินบทสรุป

### กรอบเป็นคำถามวิจัย (ไม่ใช่รายงานผลลบ)
> RQ: การใช้ material segmentation (DMS46) เสนอ region เหล็ก
> ช่วยการคัดกรองตำหนิบนภาพถ่ายไม่คุมสภาพจริงหรือไม่?

### แผนวัด
1. เก็บ `real_test/` (ภาพถ่ายมือถือ 40–60 ภาพ ตาม use case ข้อ 1 — เห็นเหล็ก + พื้นหลัง, คละมีตำหนิ/ปกติ)
2. รัน:
   ```
   python evaluate_real.py --mode both
   ```
   ได้ pipeline vs baseline: micro/macro P·R·F1 + sec/image + stage1_metal_found_rate + fallback_rate

### บทสรุปตามผล (เลือก 1 ใน 3)
| ผลที่วัดได้ | บทสรุปที่เขียน |
|---|---|
| Stage 1 เพิ่ม precision ≥ 3–5 จุด โดย recall ไม่ตก | เก็บ Stage 1 ไว้ |
| เสมอตัวเรื่อง F1 แต่ +~200 ms/ภาพ | ตัดออก — รายงานเป็น **negative ablation** (เป็น contribution ที่ให้เหตุผลกับระบบที่ง่ายกว่า) |
| ช่วยเฉพาะภาพพื้นหลังรก | เก็บแบบ **soft-gate + fallback** (implement ไว้แล้วใน `pipeline.py`: `metal_ratio < 0.05` → ตรวจทั้งภาพ) |

### หลักฐานที่มีอยู่แล้ว (`evaluate_stage1.py`, test split 416 ภาพ)
- metal_found_rate 0.35, **fallback_rate 0.78**, box_coverage 0.10, gt_area_kept 0.16, latency 208  ms/ภาพ (GPU)
- บนภาพ scene จริง (`test_images/`) DMS46 เจอเหล็ก 24–82%
- → คำทำนาย: ผลน่าจะออกทาง 2 หรือ 3 → บทสรุปที่เป็นไปได้สูงสุด
  **"DMS46 ไม่ใช่ Stage 1 ที่ดีสำหรับโดเมนนี้; classifier เหล็ก/ไม่เหล็กตัวเล็กจะเหมาะกว่า"** — ปกป้องได้

---

## 3. Baseline สำหรับเปรียบเทียบ

### 3 ชั้นบังคับ + 1 ชั้นเสริม
| ชั้น | Baseline | สถานะ | เหตุผล (ผูกกับ use case คัดกรอง) |
|---|---|---|---|
| **A** | YOLO11n ภาพเต็ม **ไม่มี Stage 1** | มีใน `evaluate_real.py --mode baseline` | **control หลักของทั้งเล่ม** — พิสูจน์ว่า 2-stage คุ้มหรือไม่ |
| **B** | train-clean vs **train-balanced** | ✅ เสร็จ (README ตารางเปรียบเทียบ) | ablation ของ class-balanced oversampling + recipe texture |
| **C** | yolo11n vs yolo11s | ✅ `train-gray-n` vs `train-gray-s` (protocol เดียวกัน) | yolo11s เพิ่มแค่ +0.017 mAP50 — model size ไม่ใช่ปัจจัยหลัก, เกน Tier 1+2 มาจาก label+gray |
| เสริม | ตัวเลขจากเปเปอร์ NEU-DET | 1 ย่อหน้า | อ้าง 2–3 ฉบับ (YOLO บน NEU-DET ปกติ mAP ~0.70–0.80) เป็นบริบท — **ไม่เทรนซ้ำ** เพราะ dataset นี้แก้ไปมาก (8 คลาส, re-split, รวม 3 แหล่ง) เทียบตรงไม่ได้ |

**ไม่ทำ** (ถ้าอาจารย์ไม่ขอ): Faster R-CNN / RT-DETR / สถาปัตยกรรมอื่น — เปลืองเวลา ไม่เกี่ยวกับโจทย์ผู้ช่วยคัดกรอง

### Metric ที่รายงาน (image-level — `evaluate_real.py` ทำให้แล้ว)
- **หลัก:** recall รายคลาส โดยเฉพาะ **rust / crack** (พลาดของอันตราย = error ที่แพงสุดใน use case นี้)
- **หลัก:** macro-F1 ทั้ง 8 คลาส (ความครอบคลุมการคัดกรอง)
- **รอง:** micro-precision (อัตราเตือนผิด → เสียเวลาคนตรวจ)
- **รอง:** latency/ภาพ (throughput การคัดกรอง + ใช้เถียงเรื่องต้นทุน Stage 1)

### คำถามที่ต้องเคลียร์กับอาจารย์
> "baseline ที่วางไว้: (A) ไม่มี Stage 1, (B) ablation augmentation [เสร็จแล้ว], (C) yolo11n vs yolo11s
> — อาจารย์อยากได้เทียบกับตัวเลขเปเปอร์ NEU-DET หรือสถาปัตยกรรม detector อื่นเพิ่มไหม"

---

## สถานะปัจจุบัน (2026-09-05)

| งาน | สถานะ |
|---|---|
| **Data leakage audit** (`check_leakage.py`) | ✅ พบ rust valid 100% / test 98% รั่ว (Roboflow burst photos) |
| **Group-aware re-split** (`resplit_grouped.py`) | ✅ split ใหม่ 3338/422/425, leakage = 0, backup `results/split_manifest_preleakagefix.json` |
| **Retrain บน split สะอาด** | ⏳ ตัวเลข mAP ทั้งหมดในตารางล่างเป็นของ split เก่า (รั่ว) — ต้องเทรน+วัดใหม่ (ดู NEXT_STEPS ขั้น 0.5) |
| Stage 2 train-balanced (yolo11n) | ✅ mAP50 0.763 / R 0.753 — baseline |
| **Stage 2 train-gray-s (yolo11s + Tier 1+2)** | ✅ **mAP50 0.853 / mAP50-95 0.537 / R 0.809** — `runs/detect/train-gray-s` (pipeline ชี้ตัวนี้), val 0.854 ≈ test → ไม่ overfit |
| Tier 1: `fix_labels.py` (merge crazing/rolled-in) + `make_grayscale_dataset.py` | ✅ |
| Tier 2: `tune_thresholds.py` → macro-F1 val 0.814 → 0.843 | ✅ `thresholds.json` |
| Ablation B (aug/oversampling) | ✅ README + `figures/confusion_compare.png` |
| `evaluate_stage1.py` | ✅ รันเต็ม 416 ภาพ → `results/stage1_dms46_test.json` |
| cross-region NMS ใน `pipeline.py` | ✅ |
| Ablation แยกผล label+gray vs 11n→11s | ✅ `train-gray-n` mAP50 0.836 vs `train-gray-s` 0.853 → label+gray = +0.29, model size = +0.017 (`results/stage2_train-gray-n.json`) |
| `real_test/` | ⏳ ต้องเก็บภาพเอง (ดู `NEXT_STEPS.md` ข้อ 3) |
| Ablation Stage 1 (`evaluate_real.py`) | ⏳ รอ `real_test/` |
| multi-seed / CI, leakage check, เทียบเปเปอร์ NEU-DET | ⏳ |
| เคลียร์ baseline กับอาจารย์ | ⏳ |
