# งานที่เหลือ — runbook

> **สถานะ 2026-09-07:** multi-seed (n=4), leakage fix, real_test (18 ภาพ), Stage 1 ablation — **เสร็จหมด**
> เหลืองานเดียวที่เป็น blocker เล่มจบ: **เคลียร์ baseline กับอาจารย์** (ขั้น 4 ข้อ 3)
> ที่เหลือเป็น optional: ขยาย real_test ให้ครบคลาส + เติม URL, และราง product (`DATA_COLLECTION.md`)
> รายละเอียดผลล่าสุดอยู่ใน `thesis_notes.md` (ตารางท้ายไฟล์) และ README หัวข้อ "Pipeline end-to-end vs Baseline"

---

## ราง product/เดโม — generalization + Stage 0 steel gate (2026-09-07)

**เป้า:** (a) ตรวจภาพเหล็กที่ไม่เคยเห็นได้ดีขึ้น (b) ภาพที่ไม่มีเหล็ก → ตอบ "ไม่พบพื้นผิวเหล็ก"

**ลองแล้วไม่เวิร์ก:** YOLO-World / COCO / DMS46 histogram zero-shot — ทั้งหมด noise ในโดเมนนี้

**ทำไปแล้ว — Stage 0 classifier "เหล็ก/ไม่เหล็ก"** (`make_gate_dataset.py` → `train_gate.py` → `steel_gate.py`):
- yolo11n-cls, positive = merged steel (ตัด crack_ = คอนกรีต), negative = DTD + Imagenette + คอนกรีต
- val top-1 0.998 แต่ **หลอกตา** — DTD/Imagenette แยกจาก lab crop ง่ายเกิน
- ของจริง: **overfit ไป lab domain** — ภาพเหล็ก scene จริง (เหล็กเส้น/ประตูสนิม) ได้ P(เหล็ก) 0.00–0.25
- `app.py` เลยใช้แบบ **advisory** — ขึ้น "ไม่พบพื้นผิวเหล็ก" เฉพาะตอน gate + DMS46 + Stage 2 เงียบพร้อมกัน
- จับได้: เอกสาร/สัตว์/ปูนฉาบ + ~ครึ่งของภาพสุ่ม ; จับไม่ได้: ภาพไม่ใช่เหล็กที่ Stage 2 หลอนว่าเจอตำหนิ

**ขั้นต่อไปถ้าจะให้ gate ใช้ได้จริง:** เก็บภาพ "เหล็ก/โลหะ scene จริง" ~600 ภาพจากเน็ต
(เหล็กเส้น, ท่อ, รั้ว, แผ่น, เครื่องมือ, สนิม — สไตล์เดียวกับ `import_labeled.py` round1)
ใส่เป็น positive แล้ว `python make_gate_dataset.py && python train_gate.py` ใหม่
→ รากปัญหาเดียวกับทั้งโปรเจค: **ขาดภาพเหล็กในสภาพใช้งานจริง**

### Round 2 — train-real2 (2026-09-07) — ผลปนเป, ยังไม่ใช้เป็น default
Import 4 dataset ใน `downloads/` (corrosion-detection-sb1, rust-detect-1350, rust-detection-small,
corrosion-and-cracks) = **1022 ภาพสนิม scene จริง** → `dataset_real/round2` → merge → fine-tune จาก
train-real1 → **`train-real2`** (OOM ตาย ~epoch 81/100)
- lab benchmark: mAP50 0.866 — **แต่หลอกตา** (merged test set มี style เดียวกับ round1/round2)
- real-photo scene rust: conf สูงขึ้น (0.23→0.51, 0.61→0.79) + localize ดีขึ้น
- **REGRESS บนสนิม lab-crop**: `rust_example.jpg` conf 0.42 → 0.04 (catastrophic-ish forgetting)
- → `app.py` ยังใช้ **train-real1** เป็น default ("ปรับโดเมน แนะนำ") ; train-real2 = "รุ่นทดลอง scene"
- `thresholds_real2.json` มี (tune lab val, rust override 0.15)
- **ยังเป็นสนิมอย่างเดียว** — 6 คลาส texture ไม่มี dataset เปิดแนว scene

**ถ้าจะทำ round 2 ให้ดีจริง:** เทรนแบบ freeze backbone / lr ต่ำ / ผสม lab crop เข้าไปด้วย
(กัน forgetting) และเทรนให้ครบ epoch ใน terminal เอง

**Round 3 ถ้าจะทำต่อ:** เทรน train-real2 ให้ครบ 100 epoch (รันใน terminal คุณเอง กัน OOM):
```powershell
python train.py --recipe camera --data merged_dataset/data_oversampled.yaml `
                --model runs/detect/train-real1/weights/best.pt --name train-real2 --resume
```
หรือหาภาพจริงของ scratches/pitted/crack-on-metal มาเพิ่ม

รันทุกคำสั่งจากโฟลเดอร์ `C:\Users\Lenovo\steel-defect-detection` โดย **activate venv ก่อน**:

```powershell
cd C:\Users\Lenovo\steel-defect-detection
.\venv\Scripts\Activate.ps1
```
(ถ้า PowerShell ไม่ยอม activate: `Set-ExecutionPolicy -Scope Process RemoteSigned` แล้วลองใหม่
หรือใช้ `.\venv\Scripts\python.exe <script>` ตรง ๆ ทุกครั้งแทน)

---

## 🔴 ขั้น 0.5 — แก้ DATA LEAKAGE (ทำ 2026-09-05, ต้อง retrain ต่อ)

**พบ:** `check_leakage.py` — rust ใน valid 100% / test 98% มีภาพเกือบเหมือนอยู่ใน train
(ชุด Roboflow "Danger-Rust" เป็นภาพถ่ายรัว → split เดิมสุ่มแยกเฟรมติดกันคนละ split)
→ rust mAP 0.995 เดิม = เฟค, overall mAP50 0.853 สูงเกินจริง

**แก้แล้ว:** `resplit_grouped.py` — group-aware stratified re-split (จับกลุ่มภาพ near-duplicate
ไว้ split เดียวกัน) กับ `merged_dataset/` + `merged_dataset_gray/` พร้อมกัน
- split ใหม่ 3338 / 422 / 425 (ครบ 8 คลาสทุก split), ratio 80/10/10 เท่าเดิม
- `check_leakage.py` หลังแก้ = **0 คู่** ทั้งสอง dataset
- backup split เดิม: `results/split_manifest_preleakagefix.json`
- `train_oversampled.txt` regenerate แล้ว (4532 บรรทัด)

**✅ เสร็จแล้ว 2026-09-06:**
- retrain `train-gray-s2` (yolo11s) + `train-gray-n2` (yolo11n) บน split สะอาด
- re-eval: `results/stage2_train-gray-{s,n}.json`, `thresholds.json`, `results/stage1_dms46_test.json`,
  `results/crossdataset_gc10.json` — อัปเดตหมด
- `pipeline.py` STAGE2_MODEL_PATH → `train-gray-s2` ; `make_figures.py` regenerate ; `test_smoke.py` ผ่าน
- ภาพตัวอย่าง demo 8 ไฟล์ใน `test_images/` แทนด้วยภาพจาก test split ใหม่
- README กล่อง ⚠️ + thesis_notes / cross_dataset_eval / literature_comparison / error_analysis อัปเดตแล้ว

**ผลลัพธ์:** overall test mAP50 0.853 → **0.840** (s2) — ตกเล็กน้อย ตัวเลขเดิมเชื่อได้หลังแก้ ;
`train-gray-n2` 0.867 ≥ s2 → model size ไม่ใช่ปัจจัยหลัก (ยืนยันชัดขึ้น)

**✅ multi-seed (ขั้น 6) — เสร็จแล้ว 2026-09-06:** `results/stage2_multiseed.json` (n=4)
mAP50 0.867 ± 0.010 ; crack อ่อนสุด 0.687 ± 0.011 ; crazing/rolled-in_scale std ~0.05
README + thesis_notes ใส่ตาราง mean ± std แล้ว. คำสั่งที่ใช้ (เก็บไว้ทำซ้ำ):

```powershell
# seed 1,2,3 (seed 0 = train-gray-n2 มีแล้ว)
foreach ($s in 1,2,3) {
  python train.py --recipe texture --data merged_dataset_gray/data_oversampled.yaml `
                  --model yolo11n.pt --name "train-gray-n2-s$s" --epochs 120 --batch 8 --patience 40 --seed $s
  python evaluate.py --mode stage2 --weights "runs/detect/train-gray-n2-s$s/weights/best.pt" `
                     --data merged_dataset_gray/data.yaml --out "results/stage2_grayn2_seed$s.json"
}
# คัดลอกผล seed 0 ให้ชื่อเข้าชุด
copy results/stage2_train-gray-n.json results/stage2_grayn2_seed0.json

# รวมเป็น mean ± std
python aggregate_seeds.py --glob "results/stage2_grayn2_seed*.json" --out results/stage2_multiseed.json
```
→ `results/stage2_multiseed.json` : mAP50 เป็น mean ± std (n=4) ต่อคลาส — ใส่ตารางในเล่มแทนตัวเลขเดียว

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

## ✅ ขั้น 1 — ปรับความแม่นยำ (Tier 1 + 2) — เสร็จแล้ว 2026-09-03

**ผล `train-gray-s`** (yolo11s, 120 ep, texture recipe, gray + label สะอาด):
mAP50 **0.853** / mAP50-95 **0.537** / R 0.809 (val 0.854 ≈ test → ไม่ overfit)
- crazing recall 0.42 → **0.80**, rolled-in_scale → 0.83
- crack ยังอ่อนสุด (mAP50 0.669) — grayscale อาจลด contrast รอยแตก
- `thresholds.json`: macro-F1 val 0.814 → **0.843**
- `pipeline.py` ชี้ `train-gray-s` แล้ว, `results/stage2_*.json` × 3 + figures อัปเดตแล้ว
- **หลักฐาน color shortcut:** train-balanced (เทรนสี) วัดบน test เทา → rust mAP 0.995 → **0.19 (R=0)**

**✅ ablation model size — เสร็จแล้ว 2026-09-04**
`train-gray-n` (yolo11n, resume จบที่ epoch 117, recipe/oversampling เดียวกับ train-gray-s):
mAP50 **0.836** / mAP50-95 0.528 / R 0.785 บน `merged_dataset_gray` test
→ เกน Tier 1+2 (train-balanced บน test เทา 0.544 → 0.836 = **+0.29**) มาจาก label สะอาด + grayscale
  ส่วน yolo11n → yolo11s เพิ่มแค่ **+0.017 mAP50** — model size ไม่ใช่ปัจจัยหลัก
- `results/stage2_train-gray-n.json`, `figures/per_class_map.png` (gray-n vs gray-s), README หัวข้อ "Ablation — model size"
```powershell
# ทำซ้ำได้ด้วย:
python train.py --recipe texture --data merged_dataset_gray/data_oversampled.yaml --name train-gray-n --epochs 120 --batch 8 --patience 40
python evaluate.py --mode stage2 --weights runs/detect/train-gray-n/weights/best.pt --data merged_dataset_gray/data.yaml --out results/stage2_train-gray-n.json
python make_figures.py --runs train-clean train-balanced --evals "train-gray-n:results/stage2_train-gray-n.json" "train-gray-s:results/stage2_train-gray-s.json"
```

<details><summary>วิธีทำเดิม (ทำไปแล้ว — เก็บไว้อ้างอิง)</summary>

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

เช็ค regression: `python test_smoke.py`

</details>

---

## ขั้น 3 — ชุดทดสอบภาพเหล็กถ่ายจริง (`real_test/`)  — ✅ รอบแรกเสร็จ (18 ภาพ)

**นี่คือสิ่งเดียวที่พิสูจน์ได้ว่า Stage 1 (DMS46) มีประโยชน์จริงหรือควรตัดทิ้ง**

> **สถานะ:** มี 18 ภาพ + labels.csv + `evaluate_real.py` รันแล้ว → ผลอยู่ใน README/`thesis_notes.md`
> สรุป: Stage 1 = negative ablation ; โมเดลเล่มจบ transfer ≈ 0 บนภาพจริง
> **optional ต่อ:** ขยายเป็น 40–60 ภาพให้ครบ 8 คลาส (ยังขาด inclusion/rolled-in_scale/crazing สิ้นเชิง)
> + เติม URL ต้นทางทุกภาพลง `real_test/SOURCES.md` ก่อนอ้างในเล่ม

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

| จากขั้น | ไฟล์ | สถานะ |
|---|---|---|
| 1 | `results/stage2_train-gray-*.json` + `thresholds.json` + multiseed | ✅ README/thesis_notes อัปเดตแล้ว |
| 3 | `results/real_before_round1.json` / `real_after_round1.json` | ✅ ตาราง Pipeline vs Baseline ใน README แล้ว |
| 4 ข้อ 1–2 | use case + Stage 1 | ✅ `thesis_notes.md` ข้อ 1–2 (Stage 1 = negative ablation) |
| **4 ข้อ 3** | **baseline ที่อาจารย์ต้องการ** | ⏳ **ยังต้องถามอาจารย์** — คำถามร่างไว้ใน `thesis_notes.md` ท้ายไฟล์ |

### สิ่งที่เหลือจริง ๆ ก่อนปิดเล่ม
1. **ถามอาจารย์เรื่อง baseline** (คำถามพร้อมใน `thesis_notes.md`) — ได้คำตอบแล้วอาจต้องรันเพิ่ม 1 อย่าง
2. (optional) ขยาย `real_test/` ให้ครบ 8 คลาส + เติม URL ต้นทาง
3. (optional) เพิ่มข้อมูล crack จากแหล่งที่ 2 แล้ว retrain 4 seeds ถ้าอยากดัน crack ขึ้นจาก 0.687
   — ไม่จำเป็นสำหรับเล่ม (รายงานเป็น limitation แล้ว)
