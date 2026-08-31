# Steel Surface Defect Detection — ระบบตรวจจับตำหนิพื้นผิวเหล็ก (2-Stage)

ตรวจจับและจำแนกตำหนิบนพื้นผิวเหล็กจากภาพถ่าย โดยแบ่งเป็น 2 ขั้นตอน

| Stage | หน้าที่ | โมเดล |
|-------|---------|-------|
| **1. Metal Localization** | หา region ในภาพที่เป็นวัสดุ "เหล็ก/โลหะ" แล้ว crop ออกมา | [DMS46](https://github.com/apple/ml-dms-dataset) (Apple Dense Material Segmentation, pre-trained, TorchScript) |
| **2. Defect Detection** | ตรวจชนิดตำหนิบนภาพที่ crop มา 8 ประเภท | YOLO11n (เทรนเองบน dataset รวม) |

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
├── train.py                 เทรนโมเดล Stage 2
├── evaluate.py              วัดผล (Stage 2 อย่างเดียว / ทั้ง pipeline)
├── merge_datasets.py        รวม NEU + Rust + Crack เป็น 8 คลาส
├── app.py                   Prototype UI (Gradio)
├── requirements.txt
├── data.yaml                config 6 คลาส NEU เดิม
├── merged_dataset/
│   ├── data.yaml            config 8 คลาส  ← ใช้เทรนจริง
│   └── {train,valid,test}/{images,labels}/
├── rust_dataset/  crack_dataset/            dataset ดิบก่อน merge
├── runs/detect/
│   ├── train/               เทรนครั้งแรก 6 คลาส (30 epochs)
│   ├── train-2/             เทรน 8 คลาส บน dataset ที่ยังปนเปื้อน (50 epochs)
│   └── train-clean/         เทรน 8 คลาส บน dataset ที่แก้แล้ว  ← pipeline ใช้ best.pt ตัวนี้
├── test_images/             ภาพตัวอย่างสำหรับทดสอบ pipeline
└── pipeline_results/        ผลลัพธ์ (generate ได้เอง)
```

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

# ปรับ threshold / บังคับใช้ CPU
python pipeline.py --image a.jpg --conf 0.35 --device cpu
```

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

เทรนเสร็จ weights อยู่ที่ `runs/detect/<name>/weights/best.pt` — ถ้าเปลี่ยนชื่อ run
ให้แก้ `STAGE2_MODEL_PATH` ใน `pipeline.py` ให้ชี้ตรง

### 3. วัดผล

```bash
python evaluate.py                 # ทั้ง Stage 2 + pipeline end-to-end
python evaluate.py --mode stage2   # เฉพาะ YOLO (mAP ต่อคลาส)
python evaluate.py --mode pipeline # เฉพาะ end-to-end (image-level P/R/F1)
```

รายงานบันทึกที่ `evaluation_results.json`

### 4. Prototype UI

```bash
pip install gradio
python app.py
# เปิด http://127.0.0.1:7860 — อัปโหลดภาพ, เลื่อน confidence, กดตรวจสอบ
```

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

> โมเดลนี้เทรนถึง epoch 40/80 แล้ว process ถูก environment หยุด (background job ถูก kill)
> ผลที่ epoch 40–45 นิ่งแล้ว (mAP50 ~0.766 บน valid) ถ้าอยากเทรนต่อให้ครบ 80
> รันในเทอร์มินัลของตัวเอง: `python train.py --name train-clean --resume`
>
> **จุดอ่อน:** `crazing` (mAP50 0.42, recall ต่ำ) — คลาสที่แยกยากด้วยตาเปล่า
> อาจต้องเพิ่มข้อมูลหรือ augmentation เฉพาะคลาสนี้

### Pipeline end-to-end (`evaluate.py --mode pipeline`)

รันบน test split image-level เทียบ "ชนิดตำหนิที่ระบบตอบ" กับ label ของแต่ละภาพ
Stage 1 (DMS46) เป็นคอขวด — ตรวจเจอ "เหล็ก" ราว 60–65% ของภาพทดสอบ (ทั้งที่เป็นภาพ
พื้นผิวเหล็กระยะใกล้) และพลาดเหล็กทาสี/สนิมหนักในภาพถ่ายจริง `pipeline.py` จึงมี
fallback: ถ้าเหล็กครอบคลุม < `--min-metal-ratio` (ค่าเริ่มต้น 5%) จะตรวจตำหนิทั้งภาพเพิ่ม

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
python merge_datasets.py        # NEU สะอาด + rust + crack (polygon -> bbox)
python resplit_dataset.py       # แบ่ง split ใหม่ให้ทุกคลาสครบทุก split
python train.py --epochs 80     # เทรนใหม่
python evaluate.py              # วัดผลซ้ำ
```

> ตัวเลขผลการทดลองด้านบนเป็นของโมเดล `train-2` (ก่อนแก้ dataset) หลังเทรนใหม่ให้
> อัปเดตหัวข้อ "ผลการทดลอง" ด้วยเลขจาก `evaluation_results.json` รอบใหม่

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
