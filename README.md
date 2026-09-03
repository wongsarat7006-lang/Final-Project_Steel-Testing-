# Steel Surface Defect Detection — ระบบตรวจจับตำหนิพื้นผิวเหล็ก (2-Stage)

ตรวจจับและจำแนกตำหนิบนพื้นผิวเหล็กจากภาพถ่าย โดยแบ่งเป็น 2 ขั้นตอน

| Stage | หน้าที่ | โมเดล |
|-------|---------|-------|
| **1. Metal Localization** | หา region ในภาพที่เป็นวัสดุ "เหล็ก/โลหะ" แล้ว crop ออกมา | [DMS46](https://github.com/apple/ml-dms-dataset) (Apple Dense Material Segmentation, pre-trained, TorchScript) |
| **2. Defect Detection** | ตรวจชนิดตำหนิบนภาพที่ crop มา 8 ประเภท | YOLO11s (เทรนเองบน dataset รวม, grayscale — `train-gray-s`) |

```
ภาพถ่าย ──▶ [Stage 1: DMS46] ──▶ mask พื้นที่เหล็ก ──▶ crop
                                                        │
        ผลลัพธ์ ◀── วาดกรอบ + ป้ายไทย ◀── [Stage 2: YOLO11] ◀──┘
```

โปรเจคนี้ใช้คู่กับ repo `ml-dms-dataset` (โค้ดต้นฉบับของ Stage 1) — ไฟล์โมเดล `DMS46_v1.pt` ถูก copy มาไว้ใน repo นี้แล้ว

---

## คลาสตำหนิ 8 ประเภท

| id | class | ชื่อไทย | ที่มา | ความเสี่ยง |
|----|-------|--------|-------|-----------|
| 0 | crazing | รอยแตกลายงา | NEU | ปานกลาง–สูง |
| 1 | inclusion | สิ่งแปลกปลอมฝังใน | NEU | ปานกลาง |
| 2 | patches | รอยแผ่น/ผิวลอก | NEU | ต่ำ–ปานกลาง |
| 3 | pitted_surface | ผิวขรุขระเป็นหลุม | NEU | ปานกลาง |
| 4 | rolled-in_scale | สะเก็ดฝังจากการรีด | NEU | ปานกลาง |
| 5 | scratches | รอยขีดข่วน | NEU | ต่ำ |
| 6 | rust | สนิม | Rust dataset (รวม RUST + DANGER-RUST) | สูง |
| 7 | crack | รอยแตกร้าว | Crack dataset | สูง |

---

## โครงสร้างโปรเจค

```
steel-defect-detection/
├── DMS46_v1.pt              โมเดล Stage 1 (TorchScript)
├── pipeline.py              รัน pipeline 2-stage เต็มระบบ  ← ไฟล์หลัก
├── train.py                 เทรนโมเดล Stage 2  (มี --recipe {default,texture})
├── evaluate.py              วัดผลบน test split (Stage 2 mAP / pipeline image-level)
├── evaluate_real.py         วัดผลบนภาพเหล็กถ่ายจริง — เทียบ pipeline vs baseline (ไม่มี Stage 1)
├── evaluate_stage1.py       วัด Stage 1 (DMS46) เชิงตัวเลข: detection rate / coverage / เวลา
├── merge_datasets.py        รวม NEU + Rust + Crack เป็น 8 คลาส
├── resplit_dataset.py       แบ่ง train/valid/test ใหม่แบบ stratified (ทุก split ครบ 8 คลาส)
├── fix_labels.py            [accuracy] รวมกล่อง crazing/rolled-in เป็น 1/ภาพ + ตัดกล่องเสีย
├── make_grayscale_dataset.py [accuracy] merged_dataset/ → merged_dataset_gray/ (ตัด shortcut สี)
├── make_oversampled_list.py class-balanced oversampling (มี --dataset) → <ds>/train_oversampled.txt
├── tune_thresholds.py       [accuracy] หา per-class confidence จาก val → thresholds.json
├── make_figures.py          สร้างรูปประกอบรายงานลง figures/
├── make_diagrams_doc.py     สร้าง docs/system_diagrams.docx (ไดอาแกรมสถาปัตยกรรม 5 รูป)
├── make_uml_doc.py          สร้าง docs/uml_sa_diagrams.docx (UML/SA 10 รูป: use case, C4, ERD, sequence, ...)
├── make_design_doc.py       สร้าง docs/design_document.docx (FR/NFR, architecture, data design, UI/UX, flowchart)
├── test_smoke.py            smoke test — รัน pipeline 1 ภาพ + เช็คโครง output (กัน regression)
├── app.py                   Prototype UI (Gradio)
├── prepare_data.md          ขั้นตอนเตรียม dataset ตั้งแต่ต้น (reproducibility)
├── data.yaml                config 6 คลาส NEU เดิม
├── data_oversampled.yaml    config 8 คลาส, train ชี้ไฟล์ oversampled list
├── thresholds.json          per-class conf (มีก็ใช้อัตโนมัติใน pipeline/app/evaluate_real)
├── results/                 ผล eval ที่เก็บไว้ (JSON) — ดู results/README.md
├── merged_dataset/          8 คลาส — labels_raw/ = label ก่อน fix_labels.py
│   ├── data.yaml            config 8 คลาส  ← ใช้เทรน baseline
│   └── {train,valid,test}/{images,labels}/
├── merged_dataset_gray/     เวอร์ชัน grayscale (สร้างจาก make_grayscale_dataset.py)
├── train/ valid/ test/      NEU ดิบจาก Roboflow (source ของ merge_datasets.py — ห้ามลบ)
├── rust_dataset/  crack_dataset/            dataset ดิบก่อน merge
├── real_test/               ★ ต้องสร้างเอง — ภาพเหล็กถ่ายจริง + labels.csv (ดู prepare_data.md ข้อ 6)
├── runs/detect/
│   ├── train/               เทรนครั้งแรก 6 คลาส (30 epochs)
│   ├── train-2/             เทรน 8 คลาส บน dataset ที่ยังปนเปื้อน (50 epochs)
│   ├── train-clean/         เทรน 8 คลาส dataset สะอาด, augmentation default (baseline)
│   └── train-balanced/      เทรน 8 คลาส + oversampling + recipe texture  ← pipeline ใช้ best.pt ตัวนี้
├── figures/                 รูปประกอบรายงาน (generate ได้เอง)
├── test_images/             ภาพตัวอย่างเล่น ๆ สำหรับ demo pipeline
└── pipeline_results/        ผลลัพธ์ (generate ได้เอง)
```

ขั้นตอนเตรียมข้อมูลแบบละเอียด (raw source → merge → resplit → oversample) อยู่ใน **`prepare_data.md`**

---

## ติดตั้ง

ต้องมี Python 3.11 และ NVIDIA GPU (แนะนำ — CPU รันได้แต่ช้า)

```bash
python -m venv venv
venv\Scripts\activate

# 1) PyTorch แบบมี CUDA (ปรับ cu124 ตาม CUDA ของเครื่อง — ดู https://pytorch.org)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2) ที่เหลือ
pip install -r requirements.txt
```

---

## วิธีใช้

### 1. รัน pipeline เต็มระบบ

```bash
# ภาพเดียว
python pipeline.py --image test_images/steel-plate3.jpg

# ทั้งโฟลเดอร์
python pipeline.py --folder test_images --output_dir pipeline_results

# ปรับ threshold / บังคับใช้ CPU / เลือกโมเดล Stage 2
python pipeline.py --image a.jpg --conf 0.35 --device cpu
python pipeline.py --image a.jpg --weights runs/detect/train-gray-s/weights/best.pt
```
ถ้ามี `thresholds.json` (จาก `tune_thresholds.py`) pipeline จะใช้ per-class conf อัตโนมัติ — ปิดด้วย `--no-class-conf`

ได้ผลลัพธ์เป็น `*_result.jpg` (ภาพ + กรอบ + ป้ายไทย) และ `*_result.json` (รายละเอียดทุก detection)
โหมด `--folder` สร้าง `_index.json` สรุปทั้งชุด

### 2. เทรนโมเดล Stage 2 ใหม่

```bash
# เตรียม dataset รวม (ครั้งเดียว)
python merge_datasets.py

# เทรน (ค่าเริ่มต้น: yolo11n, 50 epochs, batch 16 — พอดีกับ RTX 3050 6GB)
python train.py

# ปรับแต่ง
python train.py --model yolo11s.pt --epochs 100 --batch 8 --name train-s
python train.py --resume
```

เทรนเสร็จ weights อยู่ที่ `runs/detect/<name>/weights/best.pt` — ใช้กับ pipeline ได้ทันที
ด้วย `python pipeline.py --weights <path>` หรือแก้ `STAGE2_MODEL_PATH` ใน `pipeline.py` ให้เป็นค่าเริ่มต้น

### 3. วัดผล

```bash
# บน test split (NEU/Rust/Crack)
python evaluate.py --mode stage2 --weights runs/detect/<run>/weights/best.pt \
                   --out results/stage2_<run>.json     # mAP50 / mAP50-95 ต่อคลาส
python evaluate.py --mode pipeline  # end-to-end image-level P/R/F1

# บนภาพเหล็กถ่ายจริง (ต้องสร้าง real_test/ ก่อน — ดู prepare_data.md ข้อ 6)
python evaluate_real.py             # เทียบ pipeline (มี Stage 1) vs baseline (YOLO ภาพเต็ม)
python evaluate_real.py --mode pipeline --conf 0.35

# วัด Stage 1 (DMS46) เชิงตัวเลข — detection rate / coverage / ตัดตำหนิทิ้งไหม / เวลา
python evaluate_stage1.py --dir merged_dataset/test --out results/stage1_dms46_test.json
```
ผล eval ที่เก็บไว้อยู่ใน `results/` (ดู `results/README.md`)

### 4. เทรนโมเดลปรับปรุง (แก้คลาสที่อ่อน)

```bash
python make_oversampled_list.py                          # สร้าง train_oversampled.txt
python train.py --recipe texture --data data_oversampled.yaml \
                --name train-balanced --epochs 100 --batch 8 --patience 30
python evaluate.py --mode stage2                         # วัดผลใหม่
```

### 5. สร้างรูปประกอบรายงาน

```bash
python make_figures.py --runs train-clean train-balanced \
       --evals "train-clean:results/stage2_train-clean.json" "train-balanced:results/stage2_train-balanced.json"
```
ได้ `figures/{training_curves,per_class_map,class_distribution,confusion_compare}.png`

### 6. Prototype UI

```bash
pip install gradio
python app.py
# เปิด http://127.0.0.1:7860 — อัปโหลดภาพ, เลื่อน confidence, กดตรวจสอบ
```

- โหลดโมเดลครั้งเดียวตอนเริ่ม (ครั้งแรกอาจใช้เวลา ~10–20 วิ) แล้วพร้อมรับภาพ
- แสดง 3 ส่วน: สรุปผล (ชนิดตำหนิ + เน้นความเสี่ยงสูง), ภาพผลลัพธ์ (กรอบเหล็ก + กรอบตำหนิ),
  ภาพ Stage 1 (พื้นที่ที่เป็นเหล็ก / fallback ทั้งภาพ)
- ใช้ fallback + cross-region NMS แบบเดียวกับ `pipeline.py` (`pipeline.build_regions`)
- ติ๊ก "ตรวจละเอียด" = test-time augmentation (ช้าลง ~2–3x, recall ดีขึ้นเล็กน้อย)

---

## ผลการทดลอง

### Stage 2 — YOLO11n, 8 คลาส, dataset สะอาด (`runs/detect/train-clean`)

วัดบน test split ใหม่ (416 ภาพ ครบทั้ง 8 คลาส) ด้วย `evaluate.py --mode stage2 --split test`

| | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| **รวมทุกคลาส** | **0.750** | 0.448 | 0.732 | 0.703 |

| class | mAP50 | mAP50-95 |
|-------|------:|---------:|
| rust | 0.995 | 0.920 |
| scratches | 0.948 | 0.497 |
| patches | 0.897 | 0.574 |
| pitted_surface | 0.786 | 0.473 |
| inclusion | 0.738 | 0.359 |
| crack | 0.627 | 0.340 |
| rolled-in_scale | 0.585 | 0.267 |
| crazing | 0.423 | 0.159 |

เทียบกับโมเดลเดิม `train-2` (dataset ปนเปื้อน): test mAP50 0.56 → **0.75**
(ตัวเลขเดิมยังเชื่อไม่ได้เพราะ test split เก่ามีแค่ 5 คลาส)

### การพัฒนา: แก้คลาสที่โมเดล "มองข้าม" (crazing / rolled-in_scale)

จาก **confusion matrix** เทียบ train-clean vs train-balanced (`figures/confusion_compare.png`, วัดบน test split):

| คลาส | ทำนายถูก (clean → balanced) | หลุดเป็น background (clean → balanced) | สับสนข้ามคลาส |
|---|---|---|---|
| crazing | 0.36 → **0.50** | 0.64 → **0.50** | ~0 |
| rolled-in_scale | 0.66 → **0.79** | 0.34 → **0.21** | ~0 |
| inclusion | 0.76 → 0.82 | 0.24 → 0.18 | ~0 |

→ ปัญหาเดิมคือ **missed detection (recall ต่ำ) ล้วน ๆ ไม่ใช่จำผิดชนิด** สาเหตุ 2 อย่าง:

1. **Class imbalance** — crazing 555 / rolled-in_scale 504 instance เทียบกับ crack 1125 / rust 705
   (`figures/class_distribution.png`) loss ถูกครอบงำด้วยคลาสที่เยอะ
2. **Mosaic augmentation ทำลาย texture เต็มภาพ** — crazing คือร่างแหรอยแตกละเอียดกินทั้งภาพ
   200×200 px พอ mosaic นำ 4 ภาพมาต่อแล้วย่อเหลือ ~100 px + `scale=0.4` jitter
   texture ที่บ่งชี้คลาสหายไป โมเดลจึงเรียนรู้ว่า "ไม่มั่นใจ = background"

**วิธีแก้ (2 อย่างพร้อมกัน):**

| | รายละเอียด |
|---|---|
| Class-balanced oversampling | `make_oversampled_list.py` ทำซ้ำ path ในไฟล์ list (ไม่ก็อปไฟล์): crazing ×3, pitted/rolled-in_scale/scratches ×2 |
| Texture-aware augmentation | `train.py --recipe texture`: `mosaic 1.0→0.3`, `close_mosaic 10→20`, `scale 0.4→0.2`, `degrees 10→5`, `flipud 0.2→0.5`, `erasing 0.4→0.2`, `cos_lr=True` |

รันด้วย: `python train.py --recipe texture --data data_oversampled.yaml --name train-balanced --epochs 100 --batch 8`

**ผล `train-balanced`** (yolo11n, 100 epochs, texture recipe + oversampling) วัดบน test split เดียวกัน:

| | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| train-clean (baseline) | 0.750 | 0.448 | 0.732 | 0.703 |
| **train-balanced** | **0.763** | 0.449 | 0.775 | **0.753** |

รายคลาส (mAP50 / recall) — เทียบ train-clean → train-balanced:

| class | mAP50 | Δ mAP50 | recall | Δ recall |
|---|---|---|---|---|
| crazing | 0.423 → **0.392** | −0.031 | 0.258 → **0.418** | **+0.160** |
| rolled-in_scale | 0.585 → **0.668** | **+0.083** | 0.441 → **0.600** | **+0.159** |
| crack | 0.627 → 0.689 | +0.062 | 0.630 → 0.652 | +0.022 |
| inclusion | 0.738 → 0.778 | +0.040 | 0.708 → 0.751 | +0.043 |
| pitted_surface | 0.786 → 0.786 | 0.000 | 0.744 → 0.791 | +0.047 |
| scratches | 0.948 → 0.929 | −0.019 | 0.980 → 0.939 | −0.041 |
| patches | 0.897 → 0.863 | −0.034 | 0.862 → 0.874 | +0.012 |
| rust | 0.995 → 0.995 | 0.000 | 1.000 → 1.000 | 0.000 |

**อ่านผล:** recipe ทำงานตามเป้า — 2 คลาสที่ "ถูกมองข้าม" ได้ recall เพิ่มชัด
(crazing **+16 จุด**, rolled-in_scale **+16 จุด**), overall recall +5 จุด, mAP50 +1.3 จุด
- rolled-in_scale ดีขึ้นทุกทาง (mAP50 +0.083)
- crazing: recall พุ่งขึ้นแต่ precision ตก (0.65→0.53) → mAP50 แทบเท่าเดิม, mAP50-95 ตกเล็กน้อย
  (localization ยังหยาบ) — **ยังเป็นคลาสที่อ่อนสุด**
- คลาสที่แข็งอยู่แล้ว (patches, scratches) ถอยเล็กน้อย ~0.02–0.03 จากการลด mosaic/scale

**สรุป:** สำหรับงานตรวจตำหนิ (พลาดตำหนิ = แย่กว่าเตือนเกิน) recall ที่สมดุลขึ้นคุ้มกว่า mAP ที่ขยับนิด
→ train-balanced ดีกว่า train-clean. **ขั้นต่อไป (Tier 1+2) ดันได้อีกมาก — ดูหัวข้อ "Tier 1+2" ด้านล่าง**
(โมเดลหลักปัจจุบัน = `train-gray-s`, `pipeline.py` ชี้ตัวนี้แล้ว)

### ตารางเปรียบเทียบ

| Config | mAP50 | mAP50-95 | R | crazing mAP50 | rolled-in_scale mAP50 |
|---|---|---|---|---|---|
| train-2 (dataset ปนเปื้อน, test split เก่า) | 0.56 | – | – | – | – |
| train-clean (dataset สะอาด, aug default) | 0.750 | 0.448 | 0.703 | 0.423 | 0.585 |
| train-balanced (oversample + recipe texture) | 0.763 | 0.449 | 0.753 | 0.392 | 0.668 |
| **train-gray-s** (Tier 1+2 — ดูหัวข้อถัดไป) | **0.853** | **0.537** | **0.809** | **0.870** | **0.860** |

> ⚠️ **train-gray-s วัดคนละ protocol**: บน `merged_dataset_gray` test (ภาพ grayscale) และ label ของ
> crazing/rolled-in_scale ถูกรวมเป็น 1 กล่อง/ภาพ (`fix_labels.py`) — mAP ของ 2 คลาสนี้จึงวัด
> "หา region ตำหนิ" ไม่ใช่ "ระบุแต่ละ patch" เทียบตัวเลขตรง ๆ กับ 2 แถวบนไม่ได้

### Tier 1+2 — label สะอาด + grayscale + yolo11s (`runs/detect/train-gray-s`)

3 การเปลี่ยนพร้อมกัน: (1) `fix_labels.py` รวมกล่อง crazing/rolled-in_scale เป็น union ต่อภาพ +
ตัดกล่อง degenerate  (2) `make_grayscale_dataset.py` เทรน/วัดบน grayscale (ตัด shortcut เรื่องสี)
(3) yolo11n → **yolo11s**

**train-gray-s** — 120 epochs, texture recipe + oversampling, วัดบน `merged_dataset_gray` test (568 instance):

| | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| **รวมทุกคลาส** | **0.853** | **0.537** | 0.836 | 0.809 |

val mAP50 = 0.854 ≈ test 0.853 → ไม่ overfit

| class | mAP50 | mAP50-95 | P | R |
|-------|------:|---------:|---:|---:|
| rust | 0.995 | 0.974 | 0.981 | 1.000 |
| scratches | 0.964 | 0.526 | 0.921 | 0.947 |
| patches | 0.900 | 0.514 | 0.800 | 0.851 |
| crazing | 0.870 | 0.437 | 0.818 | 0.800 |
| rolled-in_scale | 0.860 | 0.565 | 0.833 | 0.832 |
| pitted_surface | 0.795 | 0.499 | 0.858 | 0.721 |
| inclusion | 0.769 | 0.373 | 0.680 | 0.660 |
| crack | 0.669 | 0.411 | 0.799 | 0.659 |

**อ่านผล:**
- คลาสที่ไม่ได้แตะ label (patches/scratches/pitted/rust/inclusion) → เสมอตัวถึงดีขึ้นเล็กน้อย
  → **grayscale ไม่ทำให้เสีย** แถมตัด color shortcut
- crazing recall 0.42 → **0.80**, rolled-in_scale recall → 0.83 — จับตำหนิได้จริงขึ้นมาก
  (ส่วน mAP ที่พุ่งแรง มาจากการเปลี่ยนรูปแบบ label ด้วย — ดู ⚠️ ข้างบน)
- **crack ยังอ่อนสุด** (mAP50 0.669, recall 0.66) — grayscale อาจลด contrast ของรอยแตกบนภาพสี

**หลักฐาน color shortcut** (วัดโมเดลที่เทรนบน**สี** บน test **เทา**):
`train-balanced` rust mAP50 **0.995 → 0.187 (recall → 0.00)** — โมเดลสีหา rust จากโทนสีเป็นหลัก
พอเป็น grayscale แทบหาไม่เจอ → เป็นเหตุผลที่ต้องเทรนบน grayscale

**ยังขาดสำหรับ ablation ที่แยกผลชัด:** retrain yolo11n บน `merged_dataset_gray` (label สะอาด + gray, ไม่เปลี่ยน model size)
→ แยกผลของ "label+gray" ออกจาก "11n→11s". ตอนนี้ทั้ง 3 อย่างรวมอยู่ใน train-gray-s

### Per-class confidence threshold (`tune_thresholds.py` → `thresholds.json`)

จาก F1-vs-confidence curve บน val — pipeline/app/evaluate_real โหลดใช้อัตโนมัติ

| | conf ที่เลือก | หมายเหตุ |
|---|---|---|
| rust | 0.90 | มั่นใจสูงตลอด ตั้งสูงได้ |
| crazing | 0.59 | |
| patches / inclusion | 0.47 / 0.45 | |
| crack / pitted / scratches / rolled-in | 0.33–0.36 | ยิงเบา ตั้งต่ำเพื่อ recall |

**macro-F1 บน val: 0.814 (conf เดียว 0.40) → 0.843 (per-class)**  (+0.029)

### Pipeline end-to-end vs Baseline

`evaluate.py --mode pipeline` (บน test split NEU/Rust/Crack) วัด image-level — แต่ภาพเหล่านี้
เป็น crop แน่นอยู่แล้ว Stage 1 จึงมักคืนกรอบ = ทั้งภาพ ผลเลย ≈ baseline

**การเทียบที่มีความหมายต้องใช้ภาพเหล็กถ่ายจริง** (`real_test/`) แล้วรัน:

```bash
python evaluate_real.py           # pipeline (Stage 1 + Stage 2) vs baseline (YOLO ภาพเต็ม)
```

| | micro-P | micro-R | micro-F1 | macro-F1 | sec/img |
|---|---|---|---|---|---|
| Pipeline (2-stage) | _?_ | _?_ | _?_ | _?_ | _?_ |
| Baseline (ไม่มี Stage 1) | _?_ | _?_ | _?_ | _?_ | _?_ |

### Stage 1 (DMS46) — metric เชิงตัวเลข (`evaluate_stage1.py`)

วัดบน `merged_dataset/test` เต็ม **416 ภาพ** (GPU) — ภาพชุดนี้เป็น NEU/Rust/Crack crop ผิวเหล็กเต็มเฟรม:

| metric | ค่า | ความหมาย |
|---|---|---|
| metal_found_rate | **0.349** | DMS46 เจอ region "Metal" บ้างใน ~1/3 ภาพ |
| fallback_rate | **0.784** | 78% ของภาพตกไป fallback = ตรวจทั้งภาพ (Stage 1 ไม่มีผล) |
| metal_ratio (mean / median) | 0.052 / **0.000** | กรอบ metal ที่เจอครอบคลุมพื้นที่จิ๋วมาก |
| box_coverage_mean | 0.103 | กรอบ metal รวมกันคลุมแค่ ~10% ของภาพ |
| gt_center_inside_rate | **0.165** | จุดกึ่งกลางกล่องตำหนิจริงตกในกรอบ metal แค่ 16% |
| gt_area_kept_mean | **0.163** | ถ้าไม่ fallback กรอบ metal จะตัดตำหนิจริงทิ้ง ~84% |
| stage1_ms (mean / median) | 208 / 228 ms (GPU) | ต้นทุนเวลาที่จ่ายเพิ่มต่อภาพ |

เทียบกับภาพถ่ายจริงระดับ scene (`test_images/`): DMS46 ตรวจเจอเหล็ก 24–82% ของภาพ
(`images.jpg` 74%, `steel-plate3.jpg` 82%)

> **สรุป:** DMS46 ทำงานเฉพาะกับภาพที่มี "บริบทฉาก" (วัตถุเหล็กอยู่ในภาพร่วมกับพื้นหลัง)
> บน close-up texture patch มันแยกไม่ออกว่าเป็นเหล็ก → ตัวเลข mAP ทั้งหมดใน `merged_dataset`
> จึงเป็น **Stage 2 ล้วน** (Stage 1 ไม่เคยทำงาน). ค่าของ Stage 1 พิสูจน์ได้เฉพาะบน `real_test/`
> ที่เป็นภาพถ่ายจริง — ว่ามันช่วยตัด false positive จากพื้นหลังที่ไม่ใช่เหล็กได้จริงหรือไม่

---

## ปัญหา dataset ที่พบและแก้แล้ว

### 1. โฟลเดอร์ NEU ปนเปื้อน (สำคัญที่สุด)

`train/valid/test` ที่ root ควรเป็น NEU 6 คลาสล้วน แต่มีภาพจาก Rust dataset
หลุดเข้ามา **~932 ไฟล์** (ชื่อ `Rust-*`, `Danger-Rust-*`, `No-Rust-*`) และถูก label
ผิดเป็น class `0` (crazing) / `1` (inclusion) ทั้งหมด → โมเดลเดิมเลยเรียนรู้ว่า
"โลหะเป็นสนิม = crazing" ทำให้ precision ของ crazing/inclusion ตกเหลือ ~0.5

**แก้แล้วใน `merge_datasets.py`:** `copy_neu_split()` รับเฉพาะไฟล์ที่ชื่อขึ้นต้นด้วย
ชื่อคลาส NEU จริง (`crazing_`, `inclusion_`, …) และกรองบรรทัด label ให้เหลือ class 0-5
→ ได้ NEU สะอาด ~1439 ไฟล์ กระจายคลาสละ ~240

### 2. test split ขาด 3 คลาส

split ทดสอบเดิมของ NEU มีแค่ crazing/inclusion/patches (ไม่มี pitted_surface,
rolled-in_scale, scratches) → **แก้ด้วย `resplit_dataset.py`** แบ่ง train/valid/test
ใหม่แบบ stratified ให้ทุก split มีครบทุกคลาส

### 3. label ของ crack เป็น polygon

`merge_datasets.py` แปลง polygon → bounding box ให้แล้ว (หมด warning "detect-segment mixed")

### ขั้นตอนสร้าง dataset ใหม่ + เทรน

```bash
python merge_datasets.py           # NEU สะอาด + rust + crack (polygon -> bbox)
python resplit_dataset.py          # แบ่ง split ใหม่ให้ทุกคลาสครบทุก split
python make_oversampled_list.py    # class-balanced list
python train.py --recipe texture --data data_oversampled.yaml --name train-balanced --epochs 100 --batch 8
python evaluate.py --mode stage2   # วัดผลซ้ำ
```

ขั้นตอนละเอียด + ที่มาข้อมูล + license → **`prepare_data.md`**

---

## ข้อจำกัด (Limitations)

1. **Domain gap ในชุดข้อมูล** — NEU-DET เป็นภาพแลปเกรย์สเกลระยะใกล้ ส่วน rust/crack เป็นภาพสี
   จาก Roboflow โมเดลอาจแยกได้ส่วนหนึ่งจาก "โทนภาพ" ไม่ใช่ลักษณะตำหนิล้วน ๆ
   (สังเกตจาก rust/scratches แม่นเกือบ 100% แต่คลาส texture ของ NEU ต่ำกว่ามาก)
2. **Stage 2 mAP วัดบน NEU crop สะอาด** ไม่ใช่ crop จริงจาก Stage 1 — ตัวเลข 0.75 เป็น
   ขอบบนของคุณภาพ component ไม่ใช่ของทั้งระบบ ต้องดู `evaluate_real.py` ประกอบ
3. **Stage 1 (DMS46) ไม่ได้เทรน/fine-tune กับโดเมนเหล็กอุตสาหกรรม** — เป็น material
   segmentation ระดับฉาก ต้องมีบริบทพื้นหลัง. บน close-up texture patch (merged_dataset test
   416 ภาพ) `evaluate_stage1.py` ได้ fallback 78%, gt_area_kept 16% →
   ตัวเลข mAP บน `merged_dataset` เป็น Stage 2 ล้วน ไม่สะท้อนทั้งระบบ
4. **ค่าของ Stage 1 ยังพิสูจน์ไม่ได้เชิงบวก** — `evaluate_stage1.py` แสดงว่ามันไม่ทำงาน
   บนชุด benchmark; ต้องมี `real_test/` (ภาพถ่ายจริง) เพื่อวัดว่าช่วยตัด false positive
   จากพื้นหลังที่ไม่ใช่เหล็กได้จริงหรือไม่
5. **crazing ยังเป็นคลาสที่ยากที่สุด** แม้หลังปรับปรุง — เป็นข้อจำกัดที่พบใน literature ของ NEU-DET เช่นกัน
6. รันทดลอง seed เดียว (seed=0) ยังไม่มีช่วงความเชื่อมั่น

---

## หมายเหตุทางเทคนิค

- **Metal class index = 22** DMS46 ทำนายเป็น index 0–45 ตามลำดับ 46 วัสดุที่โมเดลรองรับ
  วัสดุ "Metal" คือ taxonomy id 26 ซึ่งตรงกับ output index **22** (ไม่ใช่ 26 หรือ 24)
- **Preprocessing ของ Stage 1** ต้อง resize รักษาสัดส่วน (ด้านยาว = 512) แล้ว normalize
  ด้วย ImageNet mean/std แบบ scale 0–255 ตาม `ml-dms-dataset/inference.py` มิฉะนั้นผลเพี้ยน
- DMS46 คืน `tuple` ยาว 1 โดย `output[0]` มี shape `[1,1,H,W]` เป็น label map (argmax แล้วในตัวโมเดล)
- ภาษาไทยบนภาพวาดผ่าน PIL + ฟอนต์ `C:\Windows\Fonts\tahoma.ttf` (`cv2.putText` ไม่รองรับไทย)

---

## Credits

- Stage 1: [apple/ml-dms-dataset](https://github.com/apple/ml-dms-dataset) — Upchurch & Niu, *A Dense Material Segmentation Dataset for Indoor and Outdoor Scene Parsing*, ECCV 2022 (โมเดล: Apple license / dataset: CC-BY-NC 4.0)
- NEU surface defect dataset (Roboflow: pavithraa-sekar/neu-surface-defect-dataset, CC BY 4.0)
- Rust & Crack datasets (Roboflow)
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
