# real_test/ — ชุดทดสอบภาพเหล็กถ่ายจริง

ชุดนี้ใช้วัด **pipeline (Stage 1 + Stage 2) เทียบ baseline (YOLO ภาพเต็ม)** —
เป็นชุดเดียวที่พิสูจน์ได้ว่า Stage 1 (DMS46) มีประโยชน์จริงหรือควรตัดทิ้ง
(ภาพใน `merged_dataset` เป็น crop ผิวเหล็กเต็มเฟรม → Stage 1 ไม่มีงานทำ)

## ต้องเตรียมอะไร

```
real_test/
  images/        ภาพเหล็กถ่ายจริง 40–60 ภาพ  (ด้านสั้น ≥ ~1000 px)
  labels.csv     header: filename,classes   (มีไฟล์ว่างพร้อม header ให้แล้ว)
  SOURCES.md     ที่มาของแต่ละภาพ (ถ้าไม่ได้ถ่ายเอง)
```

### เงื่อนไขภาพ
- **ภาพระดับ scene** — เห็นชิ้นเหล็ก + พื้นหลัง/สภาพแวดล้อมในเฟรม
  (ไม่ใช่ crop ผิวเหล็กเต็มเฟรมแบบ NEU — แบบนั้นวัด Stage 1 ไม่ได้)
- คละกัน: **มีตำหนิ ~30–40 ภาพ** (สนิม รอยขีด รอยแตก ผิวเป็นหลุม ฯลฯ)
  + **ปกติ/ไม่มีตำหนิ ~10–20 ภาพ** (ไว้วัด false positive)
- ชื่อไฟล์ไม่มีเว้นวรรค/อักษรไทย เช่น `photo_001.jpg`

### labels.csv
```
filename,classes
photo_001.jpg,rust
photo_002.jpg,rust;scratches
photo_003.jpg,none
photo_004.jpg,pitted_surface;rust
```
- `classes` = ชนิดตำหนิที่ "เห็นในภาพ" คั่นด้วย `;` (image-level ไม่ต้องตีกรอบ)
- ไม่มีตำหนิ → `none`
- ชื่อคลาสต้องสะกดตรงนี้เท่านั้น:
  `crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches, rust, crack`
- เซฟเป็น UTF-8

## รันวัดผล

```powershell
python evaluate_real.py                     # pipeline + baseline แล้วพิมพ์ตารางเทียบ
python evaluate_real.py --mode pipeline --conf 0.35
```
ได้ `real_test_results.json` → เอาไปเติมตาราง "Pipeline vs Baseline" ใน README

รายละเอียดเพิ่มเติม: `../NEXT_STEPS.md` ข้อ 3 และ `../prepare_data.md` ข้อ 6
