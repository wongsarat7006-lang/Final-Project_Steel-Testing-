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
- `evaluate_stage1.py` — รันเต็ม 416 ภาพแล้ว → `stage1_results.json`
  (fallback 78%, gt_area_kept 16%, 208 ms/ภาพ GPU) → README หัวข้อ "Stage 1 metric เชิงตัวเลข"

---

## ขั้น 1 — เทรน `train-balanced` (งานหลัก, ~3–4 ชม.)

**ก่อนเริ่ม:** ปิดโปรแกรมกินการ์ดจอ (Chrome เยอะ ๆ, เกม, โปรแกรมตัดต่อ) — GPU มี 6GB
ถ้าเทรนแล้ว OOM ให้ลด `--batch` ลงเป็น 6 หรือ 4

```powershell
python train.py --recipe texture --data data_oversampled.yaml --name train-balanced --epochs 100 --batch 8 --patience 30
```

| อาร์กิวเมนต์ | ทำไม |
|---|---|
| `--recipe texture` | ลด mosaic/scale/rotate ที่ทำลาย texture ละเอียดของ crazing / rolled-in_scale |
| `--data data_oversampled.yaml` | ใช้ train list ที่ oversample crazing ×3, pitted/rolled/scratches ×2 |
| `--batch 8` | เพดานของการ์ด 6GB (desktop กินไปแล้ว ~1.5GB) |
| `--patience 30` | หยุดเองถ้า val ไม่ดีขึ้น 30 epoch |

**ระหว่างเทรน** — ดูใน console ทุก epoch:
- คอลัมน์ `box_loss / cls_loss / dfl_loss` ควรค่อย ๆ ลด
- `mAP50` ของ val ควรไต่ขึ้นแล้วนิ่ง ~epoch 60–80

**เทรนเสร็จ** ผลอยู่ที่ `runs\detect\train-balanced\`
- `weights\best.pt` ← โมเดลที่ดีที่สุด
- `results.png`, `confusion_matrix_normalized.png` ← ดูว่า crazing / rolled-in_scale ดีขึ้นไหม
- ตัวเลขสรุปบรรทัดสุดท้ายของ console = mAP50 / mAP50-95 บน val

**ถ้าเครื่องดับ / เผลอปิดกลางคัน** เทรนต่อได้:
```powershell
python train.py --resume --name train-balanced
```

### (ทางเลือก) เทรน yolo11s เทียบ — ทำก็ต่อเมื่อขั้น 1 หลักเสร็จแล้ว
```powershell
python train.py --recipe texture --data data_oversampled.yaml --name train-balanced-s --model yolo11s.pt --epochs 100 --batch 4 --patience 30
```

---

## ขั้น 2 — วัดผลโมเดลใหม่ (หลังขั้น 1 เสร็จ, ~5 นาที)

```powershell
# mAP50 / mAP50-95 ต่อคลาส บน test split  → เขียนทับ evaluation_results.json
python evaluate.py --mode stage2 --weights runs/detect/train-balanced/weights/best.pt

# (ถ้าเทรน yolo11s ด้วย) วัดตัวนั้นแยก ตั้งชื่อ out ไม่ให้ทับกัน
python evaluate.py --mode stage2 --weights runs/detect/train-balanced-s/weights/best.pt --out eval_balanced_s.json
```
> flag `--weights` เพิ่มให้แล้ว (เดิม `evaluate.py` ชี้ `train-clean/best.pt` แบบ hard-code)

**ส่งกลับมาให้ผม:** เนื้อหา `evaluation_results.json` + เลข mAP จาก console
→ ผมเติมตารางเปรียบเทียบใน README (แถว train-balanced) + รัน `make_figures.py` ให้

---

## ขั้น 3 — ชุดทดสอบภาพเหล็กถ่ายจริง (`real_test/`)

**นี่คือสิ่งเดียวที่พิสูจน์ได้ว่า Stage 1 (DMS46) มีประโยชน์จริงหรือควรตัดทิ้ง**

### 3.1 สร้างโฟลเดอร์
```powershell
mkdir real_test\images
```

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
| 2 | `evaluation_results.json` + เลข mAP | เติมตาราง train-balanced ใน README, รัน make_figures |
| 3 | `real_test_results.json` | เติมตาราง Pipeline vs Baseline, confusion เทียบ |
| 4 | คำตอบ 3 ข้อ | เขียนบทสรุป + ปรับ Limitations / ablation section |
