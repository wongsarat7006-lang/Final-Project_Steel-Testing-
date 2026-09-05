# Cross-dataset generalization test (GC10-DET)

แทนแผน `real_test/` 40–60 ภาพถ่ายเอง (เก็บไม่ไหว) — ใช้ **ชุดข้อมูลเหล็กสาธารณะอีกชุด
ที่โมเดลไม่เคยเทรน** เป็นตัววัด domain shift ตาม protocol มาตรฐานใน literature ของ
surface defect detection (train บน dataset A → วัดบน dataset B)

## ชุดที่ใช้: GC10-DET

- ผิวเหล็กแผ่นจาก **สายการผลิตจริง** 10 ชนิดตำหนิ (Lv et al., GC10-DET)
- ใช้ test split 229 ภาพ (ดาวน์โหลดได้ 224) จาก Roboflow export บน HuggingFace
  `imaadd05/gc10-det` — License CC-BY-4.0
- อยู่ที่ `external_test/gc10/` (gitignored — โหลดใหม่ด้วย `scratchpad/dl_gc10.py` หรือ HF)

### taxonomy ต่างกัน → map ได้คลาสเดียวชัด ๆ
GC10 (pinyin): chongkong เจาะรู · hanfeng แนวเชื่อม · yueyawan รอยเว้า · shuiban คราบน้ำ ·
youban คราบน้ำมัน · siban คราบไหม · **yiwu = inclusion** · yahen รอยกดจากรีด ·
zhehen รอยพับ · yaozhed รอยงอเอว

→ map ตรงได้แค่ `7_yiwu → inclusion` (`external_test/gc10/classmap.json`)
คลาสอื่นวัดได้เฉพาะ **class-agnostic localization** (pred box ใด ๆ ทับ GT box ไหม)

## วิธีรัน

```
python evaluate_cross_dataset.py --dir external_test/gc10 --map external_test/gc10/classmap.json
```
วัด 3 อย่าง เทียบ pipeline (Stage1+2) vs baseline (Stage2 ภาพเต็ม):
1. **presence recall** — ภาพมีตำหนิ ระบบตรวจเจอ ≥1 กล่องไหม
2. **loc-agnostic** recall/precision — จับคู่กล่องที่ IoU ≥ 0.5 ไม่สนคลาส
3. **mapped** — inclusion: P/R/F1 ระดับกล่อง (IoU ≥ 0.5 + คลาสตรง)

## ผล (โมเดล `train-gray-s` split เก่า — ต้องรันซ้ำหลัง retrain)

`results/crossdataset_gc10.json` · 224 ภาพ · 347 GT boxes

| metric | pipeline (S1+S2) | baseline (S2 ภาพเต็ม) |
|---|---|---|
| presence recall | 0.237 | 0.138 |
| loc-agnostic recall | **0.009** | **0.009** |
| loc-agnostic precision | 0.050 | 0.094 |
| inclusion (mapped) TP/FP/FN | 0 / 14 / 34 | 0 / 10 / 34 |
| Stage 1 metal-found rate | 0.46 | — |
| Stage 1 fallback rate | 0.78 | — |

### สรุป
- **โมเดลแทบไม่ transfer ไป GC10-DET** — loc-agnostic recall ~1% (จับคู่ได้ 3/347 กล่อง),
  inclusion ตรวจถูกตำแหน่ง 0 ครั้ง (ทายมั่ว 14 ครั้ง) — สอดคล้องกับ EA-1 (domain gap)
- pipeline presence recall > baseline (+0.10) เพราะ region fallback ทำให้ยิงกล่อง (ที่ส่วนใหญ่ผิด)
  มากขึ้น → precision ตกลง — **Stage 1 ไม่ได้ช่วยความแม่นบนชุดนี้**
- ยืนยันว่าตัวเลข benchmark (mAP ~0.85 บน NEU-style crop) **ไม่ generalize** ไปโดเมนเหล็กอื่น
  → เป็นข้อจำกัดหลักของระบบที่ต้องระบุในบทสรุป

### ทำต่อ
- [ ] รันซ้ำด้วยโมเดลที่ retrain บน split สะอาด (คาดว่าผลเชิงคุณภาพไม่เปลี่ยน)
- [ ] (ถ้ามีเวลา) เพิ่ม Severstal steel defect (`Voxel51/severstal_steel_defects` บน HF) เป็น probe ที่ 2
      — เป็น segmentation mask 4 คลาสไม่มีชื่อ ต้องแปลงเป็น bbox ก่อน
