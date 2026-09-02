# งานที่เหลือ — runbook

รันทุกคำสั่งจากโฟลเดอร์ `C:\Users\Lenovo\steel-defect-detection` โดย **activate venv ก่อน**:

```powershell
cd C:\Users\Lenovo\steel-defect-detection
.\venv\Scripts\Activate.ps1
```
(ถ้า PowerShell ไม่ยอม activate: `Set-ExecutionPolicy -Scope Process RemoteSigned` แล้วลองใหม่
หรือใช้ `.\venv\Scripts\python.exe <script>` ตรง ๆ ทุกครั้งแทน)

---

## ✅ ขั้น 0 — เสร็จแล้ว (ผมทำให้)

- cross-region NMS ใน `pipeline.py` (+ flag `--nms-iou`)
- `evaluate_stage1.py` — รันเต็ม 416 ภาพแล้ว → `results/stage1_dms46_test.json`
  (fallback 78%, gt_area_kept 16%, 208 ms/ภาพ GPU) → README หัวข้อ "Stage 1 metric เชิงตัวเลข"
- **รวมตรรกะ fallback เป็นฟังก์ชันเดียว** `pipeline.build_regions()` — ใช้ร่วมกันโดย
  `pipeline.py` / `evaluate.py` / `evaluate_real.py` / `app.py` (เดิม copy กัน 4 ที่ + drift)
- **`app.py` เพิ่ม fallback + cross-region NMS** — เดิม UI ตอบ "ไม่พบเหล็ก → ไม่ตรวจ"
  เมื่อ DMS46 เจอเหล็กน้อย (เกิดบ่อยมากกับภาพจริง) ตอนนี้ตรวจทั้งภาพเผื่อเหมือน `pipeline.py`
- **`evaluate.py --mode pipeline`** ใช้ fallback แบบเดียวกับ pipeline จริงแล้ว (เดิม fallback เฉพาะตอนไม่เจอกรอบเลย)
- **สร้างโครง `real_test/`** ไว้แล้ว (`images/`, `labels.csv` header, `SOURCES.md`, `README.md`) — แค่ใส่ภาพ + เติม labels.csv
- **สคริปต์ปรับความแม่นยำ Tier 1+2** เขียน + ทดสอบแล้ว (ดูขั้น 1 ข้างล่าง):
  `fix_labels.py`, `make_grayscale_dataset.py`, `make_oversampled_list.py --dataset`,
  `tune_thresholds.py` + per-class conf ต่อสายเข้า `pipeline.py`/`app.py`/`evaluate_real.py` แล้ว

---

## ขั้น 1 — ปรับความแม่นยำ (Tier 1 + 2)

หลักฐาน (`train-balanced/results.csv` + label geometry):
- val mAP50 พีค **epoch 62 (0.80)** แล้วไหลลง → เทรนนานขึ้น/imgsz สูงขึ้น **ไม่ช่วย**
- crazing/rolled-in_scale annotate เป็นกล่องย่อยมั่ว (เฉลี่ย 2–5 กล่อง) → mAP ตันที่ label noise
- rust/scratches เกือบ 100% ส่วนหนึ่งเพราะโมเดลอ่าน "โทนสี" แยก NEU (เทา) vs rust/crack (สี)

### 1.1 เตรียม dataset ใหม่ (Tier 1 — ~10 นาที)

```powershell
python merge_datasets.py            # (ถ้ายังไม่ได้ทำ)
python resplit_dataset.py           # (ถ้ายังไม่ได้ทำ)
python fix_labels.py                # รวมกล่อง crazing/rolled-in เป็น 1 กล่อง/ภาพ + ตัดกล่องเสีย
python make_grayscale_dataset.py    # -> merged_dataset_gray/  (ตัด shortcut เรื่องสี)
python make_oversampled_list.py --dataset merged_dataset_gray
```

> `fix_labels.py` สำรองของเดิมไว้ที่ `merged_dataset/<split>/labels_raw/` — คืนค่าได้ด้วย `python fix_labels.py --restore`

### 1.2 เทรน yolo11s บน dataset ใหม่ (Tier 2 — ~3–4 ชม.)

**ก่อนเริ่ม:** ปิดโปรแกรมกินการ์ดจอ — GPU 6GB. OOM ให้ลด `--batch` เป็น 4

```powershell
python train.py --recipe texture --data merged_dataset_gray/data_oversampled.yaml `
                --model yolo11s.pt --name train-gray-s --epochs 120 --batch 6 --patience 40
```
`yolo11s.pt` จะโหลดเองอัตโนมัติครั้งแรก. เครื่องดับกลางคัน: `python train.py --resume --name train-gray-s`

> อยากแยกผลของแต่ละ fix: เทรน `--data merged_dataset/data_oversampled.yaml` (ไม่ gray) เป็น ablation ด้วย
> — ต้องรัน `make_oversampled_list.py --dataset merged_dataset` ก่อน

### 1.3 วัดผล + หา per-class threshold (~10 นาที)

```powershell
# mAP50 / mAP50-95 ต่อคลาส บน test  -> results/stage2_train-gray-s.json
python evaluate.py --mode stage2 --weights runs/detect/train-gray-s/weights/best.pt --data merged_dataset_gray/data.yaml --out results/stage2_train-gray-s.json

# per-class confidence threshold จาก val  -> thresholds.json (pipeline/app จะใช้เอง)
python tune_thresholds.py --weights runs/detect/train-gray-s/weights/best.pt --data merged_dataset_gray/data.yaml

# ใช้โมเดลใหม่กับ pipeline: python pipeline.py --weights runs/detect/train-gray-s/weights/best.pt ...
#   หรือแก้ STAGE2_MODEL_PATH ใน pipeline.py ให้เป็น default
```

> เทียบให้ยุติธรรม: วัด train-clean / train-balanced ใหม่บน **test label ชุดเดียวกัน** ด้วย
> `python evaluate.py --mode stage2 --weights runs/detect/train-clean/weights/best.pt --data merged_dataset_gray/data.yaml --out results/stage2_train-clean.json` (และ train-balanced)
> — ตัวเลขเดิมใน README วัดก่อน `fix_labels.py` เทียบตรงไม่ได้

**ส่งกลับมา:** `results/stage2_train-gray-s.json` + `thresholds.json` + เลข mAP จาก console
→ เติมตารางเทียบใน README + รัน `make_figures.py`

เช็ค regression ก่อนส่ง: `python test_smoke.py`

---

## ขั้น 3 — ชุดทดสอบภาพเหล็กถ่ายจริง (`real_test/`)

**นี่คือสิ่งเดียวที่พิสูจน์ได้ว่า Stage 1 (DMS46) มีประโยชน์จริงหรือควรตัดทิ้ง**

### 3.1 โฟลเดอร์
`real_test\images\` + `real_test\labels.csv` (header) มีให้แล้ว — ดู `real_test\README.md`

### 3.2 หาภาพ 40–60 ภาพ ใส่ `real_test\images\`
เงื่อนไขภาพ:
- **ภาพถ่ายจริงระดับ scene** — เห็นชิ้นเหล็ก + มีพื้นหลัง/สภาพแวดล้อมในเฟรม
  (ไม่ใช่ crop ผิวเหล็กเต็มเฟรมแบบ NEU — แบบนั้นวัด Stage 1 ไม่ได้)
- ด้านสั้นอย่างน้อย ~1000 px
- คละกัน: **มีตำหนิ ~30–40 ภาพ** (สนิม, รอยขีด, รอยแตก, ผิวเป็นหลุม ฯลฯ)
  + **ปกติ/ไม่มีตำหนิ ~10–20 ภาพ** (ไว้วัด false positive)
- ตั้งชื่อไฟล์เรียบ ๆ `photo_001.jpg` ... หรือชื่ออะไรก็ได้ที่ไม่มีเว้นวรรค/อักษรไทย

แหล่งภาพ: ถ่ายเอง (ดีสุด) / โรงงาน-อู่-ร้านเหล็กแถวบ้าน / เหล็กเป็นสนิมรอบตัว /
ถ้าจำเป็นดึงจากเน็ตได้แต่ต้องจดที่มาลง `real_test\SOURCES.md`

### 3.3 ทำไฟล์ label `real_test\labels.csv`
เปิด Excel / Notepad สร้างไฟล์ (บรรทัดแรกคือ header เป๊ะ ๆ):

```
filename,classes
photo_001.jpg,rust
photo_002.jpg,rust;scratches
photo_003.jpg,none
photo_004.jpg,pitted_surface;rust
```

กติกา:
- `classes` = ชนิดตำหนิที่ "เห็นในภาพ" คั่นด้วย `;` (image-level ไม่ต้องตีกรอบ)
- ไม่มีตำหนิ → ใส่ `none`
- ชื่อคลาสต้องสะกดตรงนี้เท่านั้น:
  `crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches, rust, crack`
- เซฟเป็น UTF-8

### 3.4 รันวัดผล
```powershell
python evaluate_real.py            # ทำทั้ง pipeline + baseline แล้วพิมพ์ตารางเทียบ
```
ได้ `real_test_results.json`

**ส่งกลับมาให้ผม:** `real_test_results.json` → ผมเติมตาราง "Pipeline vs Baseline" ใน README
+ รูป confusion เทียบ

---

## ขั้น 4 — ตอบ 3 คำถาม (ไว้เขียนบทสรุป / ตอบอาจารย์)

1. **use case จริง** ของระบบนี้คืออะไร — ตรวจเหล็กเส้น/แผ่นในโรงงาน? งานตรวจสภาพโครงสร้าง?
   คัดของเข้าคลัง? (มีผลต่อว่า metric ไหนสำคัญ — recall ของ crack/rust vs precision รวม)

2. **จะเก็บ Stage 1 ไว้ หรือทำ ablation ตัดทิ้ง** — จากผล `evaluate_stage1.py` ตอนนี้
   DMS46 fallback 78% และตัดตำหนิจริงทิ้ง 84% เมื่อไม่ fallback
   ทางเลือก: (ก) เก็บไว้ แล้วพิสูจน์ด้วย real_test ว่าช่วยตัด false positive จากพื้นหลัง
   (ข) ตัดทิ้ง เหลือ YOLO ภาพเต็ม แล้วรายงานเป็น ablation ว่า "ลองแล้วไม่คุ้ม"

3. **อาจารย์อยากได้ baseline อะไรเทียบ** — YOLO ภาพเต็มอย่างเดียว? / เทียบกับเปเปอร์ NEU-DET
   ที่มี mAP รายงานไว้? / เทียบ yolo11n vs yolo11s vs รุ่นอื่น?

---

## สรุปสิ่งที่ต้องส่งกลับมา

| จากขั้น | ไฟล์ | ผมจะทำต่อ |
|---|---|---|
| 1 | `results/stage2_train-gray-s.json` + `thresholds.json` + เลข mAP | เติมตารางเทียบใน README (Tier 1/2), รัน make_figures |
| 3 | `real_test_results.json` | เติมตาราง Pipeline vs Baseline, confusion เทียบ |
| 4 | คำตอบ 3 ข้อ | เขียนบทสรุป + ปรับ Limitations / ablation section |
