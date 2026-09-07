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

**→ ผลจริง (2026-09-07, `real_test/` 18 ภาพ): แถวที่ 2 — negative ablation.**
Stage 1 ทำงานบ่อยขึ้นบนภาพ scene จริง (metal_found_rate 0.35 บน crop แล็บ → **0.56** บน real_test)
แต่ pipeline (มี Stage 1) micro-P **ต่ำกว่า** baseline (ไม่มี Stage 1) ทั้ง train-gray-n2 (0.143 vs 0.333)
และ train-real1 (0.692 vs 0.750) — กรอบ metal ที่เพี้ยน + fallback ทำให้เกิด FP เพิ่มโดยไม่ได้ recall กลับมา
เก็บ `pipeline.py` (soft-gate + fallback + cross-region NMS) ไว้เป็นโค้ด แต่บทสรุปเล่ม = **"2-stage ไม่คุ้มสำหรับ use case นี้;
material segmentation ระดับฉากไม่เหมาะเป็น front-end — future work คือ classifier เหล็ก/ไม่เหล็กตัวเล็ก fine-tune เอง"**

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
| **C** | yolo11n vs yolo11s | ✅ `train-gray-n2` vs `train-gray-s2` (split สะอาด, protocol เดียวกัน) | บน split สะอาด yolo11n **สูงกว่า** yolo11s เล็กน้อย (0.867 vs 0.840) — ยืนยันชัดว่า model size ไม่ใช่ปัจจัยหลัก |
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
| **Data leakage audit** (`check_leakage.py`) | ✅ พบ rust valid 100% / test 98% รั่ว (Roboflow burst photos) 932 คู่ |
| **Group-aware re-split** (`resplit_grouped.py`) | ✅ split ใหม่ 3338/422/425, leakage = 0, backup `results/split_manifest_preleakagefix.json` |
| **Retrain + re-eval + multi-seed บน split สะอาด** | ✅ 2026-09-06 — **โมเดลหลัก = `train-gray-n2` (yolo11n)** test mAP50 **0.867 ± 0.010** (n=4); yolo11s 0.840 (ต่ำกว่า); ตัวเลขในตารางล่างอัปเดตแล้ว |
| Stage 2 train-balanced (yolo11n) | ✅ mAP50 0.763 / R 0.753 — baseline |
| **Stage 2 train-gray-n2 (yolo11n, split สะอาด) — โมเดลหลัก** | ✅ **test mAP50 0.867 ± 0.010 / mAP50-95 0.536 ± 0.009 / R 0.809** (multi-seed n=4) — `runs/detect/train-gray-n2` (pipeline ชี้ตัวนี้) |
| Stage 2 train-gray-s2 (yolo11s, split สะอาด) | ✅ test mAP50 0.840 — เล็กกว่า n2 ~2.7σ; ใหญ่กว่า 4x, ช้ากว่า 2x → ไม่ใช้ |
| multi-seed (`aggregate_seeds.py`, n=4) | ✅ `results/stage2_multiseed.json` — crack อ่อนสุด 0.687 ± 0.011; crazing/rolled-in_scale std สูง (~0.05) |
| Tier 1: `fix_labels.py` (merge crazing/rolled-in) + `make_grayscale_dataset.py` | ✅ |
| Tier 2: `tune_thresholds.py` (n2) → macro-F1 val 0.824 → 0.844 | ✅ `thresholds.json` |
| Ablation B (aug/oversampling) | ✅ README + `figures/confusion_compare.png` |
| `evaluate_stage1.py` | ✅ รันเต็ม 416 ภาพ → `results/stage1_dms46_test.json` |
| cross-region NMS ใน `pipeline.py` | ✅ |
| Ablation แยกผล label+gray vs 11n vs 11s | ✅ split สะอาด: `train-gray-n2` **0.867 ± 0.010** > `train-gray-s2` 0.840 → **yolo11n ดีกว่า** yolo11s; เกนหลักมาจาก label+gray |
| `real_test/` | ⚠️ มี 18 ภาพ (net-sourced, เอียง rust, ไม่มี inclusion/rolled-in/crazing, ยังไม่มี URL) — ใช้ได้ระดับ "แสดง domain gap" ไม่ใช่ validation เต็ม |
| Ablation Stage 1 (`evaluate_real.py`) | ✅ 2026-09-07 — **negative ablation**: บน real_test Stage 1 metal_found_rate 0.56 แต่ pipeline micro-P < baseline ทั้ง train-gray-n2 และ train-real1 → เก็บโค้ดไว้ รายงานว่าไม่คุ้ม (บทสรุป = แถวที่ 2 ในตารางข้อ 2) |
| Real-photo domain gap | ✅ train-gray-n2 บน real_test: micro-R 0.043, rust 0/12 (conf 0.92 แล็บกรองหมด) — ตรงกับ GC10 transfer ≈ 0. ราง future work `train-real1` (+672 ภาพ corrosion): rust recall 0/12 → 8/12, micro-F1 0.067 → 0.50 |
| multi-seed (n=4), leakage check, เทียบเปเปอร์ NEU-DET | ✅ `results/stage2_multiseed.json`, `check_leakage.py`, `literature_comparison.md` |
| cross-dataset test (GC10-DET แทน real_test) | ✅ `evaluate_cross_dataset.py`, `cross_dataset_eval.md` — transfer ~0 |
| Error Analysis | ✅ `docs/error_analysis.docx` (EA-1..5) |
| เคลียร์ baseline กับอาจารย์ | ⏳ |
