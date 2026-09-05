# การเทียบกับ literature (NEU-DET / GC10-DET)

ไว้เขียนบท "งานที่เกี่ยวข้อง" และ "การอภิปรายผล" — **ใช้เป็นบริบท/ตรวจความสมเหตุสมผล
ไม่ใช่การเทียบตัวต่อตัว** เพราะ dataset ของเราต่างจากงานอ้างอิง (ดูหัวข้อ "ทำไมเทียบตรงไม่ได้")

## 1. ตัวเลข mAP@0.5 บน NEU-DET จาก literature (YOLO family)

| โมเดล | NEU-DET mAP@0.5 | ที่มา |
|---|---|---|
| Faster R-CNN (baseline ดั้งเดิม) | ~0.76 | He et al. 2020 (IEEE TIM) |
| DDN (He et al. 2020) | ~0.82 | He et al. 2020 |
| YOLOv5s | 0.74–0.76 | หลายงาน; YOLOv5s-CoordAtt 0.763 |
| YOLOv8n | 0.736–0.759 | SLF-YOLO / SCCI-YOLO papers 2025 |
| YOLOv8l | 0.741 | Ashrafi et al. 2026 (edge benchmark) |
| YOLO11n | ~0.78 | YOLO-MFD paper (บอกดีขึ้น 5.9% เป็น 0.838) |
| YOLOv8 + attention/multiscale (ปรับปรุง) | 0.78–0.86 | ECM-YOLO 0.789, SLF-YOLO 0.80, SCCI-YOLO 0.786, YOLO-MFD 0.838, MBDNet-Attn-YOLO 0.858 |

**สรุปช่วง:** YOLO มาตรฐานบน NEU-DET ปกติ **0.73–0.79**; รุ่นปรับปรุง (attention / multi-scale
feature fusion / loss ใหม่) ดันขึ้น **0.78–0.86**

## 2. GC10-DET

- Lv et al. 2020, "Deep Metallic Surface Defect Detection: The New Benchmark and
  Detection Network", *Sensors* 20(6):1562 — 2,300 ภาพ, 10 คลาส, สายการผลิตจริง
- ยอมรับกันว่า **ยากกว่า NEU-DET** (ตำหนิหลากหลาย, สภาพแสง/พื้นผิวไม่คุม)
- โมเดลที่ **เทรนบน GC10-DET** โดยตรง ทำได้ราว mAP 0.60–0.75 (ต่ำกว่า NEU-DET ชัดเจน)

## 3. ผลของโปรเจคนี้ (วางในบริบท)

| | ค่า | หมายเหตุ |
|---|---|---|
| Stage 2 (train-gray-s) บน benchmark ของเรา | mAP50 ~0.82–0.85 (รอ retrain บน split สะอาด) | อยู่ในช่วงบนของ YOLO มาตรฐาน / ล่างของรุ่นปรับปรุง |
| Cross-dataset → GC10-DET (train NEU-style, test GC10) | loc-agnostic recall ~0.01 | ดู `cross_dataset_eval.md` — **transfer แทบเป็นศูนย์** |

## 4. ทำไมเทียบตรงกับ literature ไม่ได้ (ต้องเขียนกำกับไว้)

1. **จำนวนคลาสต่าง** — เรารวม 3 แหล่งเป็น 8 คลาส (NEU 6 + rust + crack จาก Roboflow)
   งานอ้างอิงเกือบทั้งหมดใช้ NEU 6 คลาสล้วน
2. **grayscale** — เราแปลงทั้ง dataset เป็น grayscale เพื่อตัด color shortcut (rust/crack เดิมเป็นภาพสี)
   → task ต่างจาก NEU-DET ต้นฉบับ
3. **re-split เอง** — group-aware 80/10/10 (ดู `resplit_grouped.py`); test set คนละชุดกับงานอื่น
4. **label ผ่าน `fix_labels.py`** — รวมกล่อง crazing/rolled-in ที่ annotate ย่อยมั่ว
   → ตัวเลขก่อน/หลังเทียบกันเองไม่ได้ ยิ่งเทียบข้ามงานไม่ได้

→ วิธีเขียนที่ถูกต้อง: "ผลของเราสอดคล้องกับช่วงที่รายงานใน literature สำหรับ YOLO บน NEU-DET
(0.73–0.86) ซึ่งยืนยันว่า pipeline ทำงานถูกต้อง แต่ไม่อ้างว่าดีกว่า/แย่กว่างานใดเพราะ setup ต่างกัน"

## 5. อ้างอิง (ตรวจ/เติม bibliographic ให้ครบก่อนใส่เล่ม)

- He, Y., Song, K., Meng, Q., Yan, Y. (2020). *An End-to-End Steel Surface Defect
  Detection Approach via Fusing Multiple Hierarchical Features.* IEEE Transactions on
  Instrumentation and Measurement, 69(4), 1493–1504. (NEU-DET detection benchmark, DDN)
- Song, K., Yan, Y. (2013). *A noise robust method based on completed local binary
  patterns for hot-rolled steel strip surface defects.* Applied Surface Science. (NEU database ต้นทาง)
- Lv, X., Duan, F., Jiang, J., Fu, X., Gan, L. (2020). *Deep Metallic Surface Defect
  Detection: The New Benchmark and Detection Network.* Sensors, 20(6), 1562. (GC10-DET)
- Jocher, G. et al. *Ultralytics YOLO* (v5/v8/v11) — https://github.com/ultralytics/ultralytics
- (รุ่นปรับปรุงที่อ้างช่วงตัวเลข 2025–2026): SLF-YOLO (Sci Rep), ECM-YOLO (JOSA A),
  SCCI-YOLO (Sci Rep), YOLO-MFD (Sci Rep), MBDNet-Attention-YOLO (Sensors/PMC) —
  ใส่เฉพาะที่จะอ้างจริง 2–3 ฉบับพอ

> ตัวเลขในตารางข้อ 1 มาจากการสำรวจ literature (ก.ย. 2026) — **ผู้เขียนต้องเปิดเปเปอร์ยืนยัน
> ตัวเลขและหน้าอ้างอิงก่อนใส่เล่ม** อย่าอ้างต่อจากไฟล์นี้ตรง ๆ
