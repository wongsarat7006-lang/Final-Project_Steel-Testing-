# การเตรียมข้อมูล (Data Preparation)

เอกสารนี้อธิบายขั้นตอนสร้าง dataset สำหรับ Stage 2 ตั้งแต่ต้นจนพร้อมเทรน
เพื่อให้ทำซ้ำผลได้ (reproducibility)

## ภาพรวม

```
NEU-DET (6 คลาส)  ─┐
Rust dataset       ─┼─►  merge_datasets.py  ─►  merged_dataset/ (8 คลาส, split เดิม)
Crack dataset      ─┘                              │
                                                   ▼
                                          resplit_dataset.py   (แบ่ง stratified ใหม่)
                                                   │
                                                   ▼
                                       make_oversampled_list.py (class-balanced list)
                                                   │
                                                   ▼
                              data_oversampled.yaml  ─►  train.py
```

## 1. แหล่งข้อมูลดิบ (raw sources)

| โฟลเดอร์ | ที่มา | คลาสเดิม | หมายเหตุ |
|---|---|---|---|
| `train/ valid/ test/` (ที่ root) | NEU Surface Defect — Roboflow (`pavithraa-sekar/neu-surface-defect-dataset` v1, CC BY 4.0) | crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches | **ปนภาพ Rust ที่ label ผิดเป็น class 0/1 ~932 ไฟล์** — `merge_datasets.py` กรองด้วย prefix ชื่อไฟล์ |
| `rust_dataset/` | Roboflow (RUST / DANGER-RUST / NO-RUST) | 0=DANGER-RUST, 1=NO-RUST, 2=RUST | ตัด NO-RUST ทิ้ง, รวม DANGER-RUST + RUST → คลาสเดียว |
| `crack_dataset/` | Roboflow (crack) | 0=crack | มีเยอะเกินสมดุล → `merge_datasets.py` cap ไว้ (train 800 / valid 150 / test 150) |

> ทั้งสามโฟลเดอร์อยู่ใน `.gitignore` (ไฟล์ใหญ่) — ถ้า clone ใหม่ต้องดาวน์โหลดเองจาก Roboflow แล้ววางตามโครงข้างบน
> ที่มาฉบับเต็ม + license อยู่ใน `README.dataset.txt` / `README.roboflow.txt` ของแต่ละโฟลเดอร์

## 2. รวม 3 dataset เป็น 8 คลาส

```bash
python merge_datasets.py
```

ได้ `merged_dataset/` โครงสร้าง `{train,valid,test}/{images,labels}` + `data.yaml`

การจับคู่คลาสใหม่:

| id | ชื่อ | มาจาก |
|---|---|---|
| 0 | crazing | NEU |
| 1 | inclusion | NEU |
| 2 | patches | NEU |
| 3 | pitted_surface | NEU |
| 4 | rolled-in_scale | NEU |
| 5 | scratches | NEU |
| 6 | rust | Rust dataset (DANGER-RUST + RUST) |
| 7 | crack | Crack dataset |

กลไกกันข้อมูลปนเปื้อน:
- รับเฉพาะไฟล์ NEU ที่ชื่อขึ้นต้นด้วยชื่อคลาสจริง (`crazing_`, `inclusion_`, ...) — `NEU_VALID_PREFIXES`
- เก็บเฉพาะบรรทัด label ที่เป็น class 0–5 ในไฟล์ NEU
- แปลง polygon/segment → bounding box (กัน dataset ผสม detect + segment)

## 3. แบ่ง split ใหม่แบบ stratified

split เดิมจาก Roboflow ไม่สมดุล (บาง split คลาสไม่ครบ 8) จึงรวมทั้งหมดแล้วแบ่งใหม่:

```bash
python resplit_dataset.py
```

ผล: ทุก split มีครบ 8 คลาส — train 3353 / valid 416 / test 416 (seed=0)

การกระจาย instance ต่อคลาส (train):

| คลาส | images | instances |
|---|---|---|
| crazing | 240 | 555 |
| inclusion | 304 | 783 |
| patches | 275 | 701 |
| pitted_surface | 241 | 349 |
| rolled-in_scale | 240 | 504 |
| scratches | 241 | 444 |
| rust | 705 | 705 |
| crack | 849 | 1125 |

> มี label ไฟล์ที่มีหลายคลาสในภาพเดียว ~97 ไฟล์ (ส่วนใหญ่จาก NEU-DET ที่ภาพหนึ่งมีตำหนิ 2 ชนิด) — ปล่อยไว้ตามจริง

## 3.5 ทำความสะอาด label + ตัด domain shortcut (accuracy — Tier 1)

```bash
python fix_labels.py                # รวมกล่อง crazing(0)/rolled-in_scale(4) เป็น 1 กล่อง/ภาพ + ตัดกล่องเสีย
python make_grayscale_dataset.py    # merged_dataset/ -> merged_dataset_gray/ (grayscale 3ch)
```

- **`fix_labels.py`** — NEU-DET annotate crazing/rolled-in_scale เป็นกล่องย่อยกระจาย (เฉลี่ย 2–5 กล่อง,
  สัดส่วนไม่คงที่) ทำให้ detector เรียนไม่ได้ → mAP50 ตันที่ ~0.39. ทั้งสองคลาสเป็น texture ทั้ง patch
  จึงรวมเป็น union box เดียว. ตัดกล่อง degenerate (w/h < 0.004 หรือ area < 0.0005) ด้วย (~19 กล่องใน crack).
  ต้นฉบับสำรองที่ `merged_dataset/<split>/labels_raw/` — คืนด้วย `python fix_labels.py --restore`
- **`make_grayscale_dataset.py`** — rust/scratches แม่นเกือบ 100% ส่วนหนึ่งเพราะโมเดลแยก
  "NEU (เทา) vs rust/crack (สี)" จากโทนสี ไม่ใช่ลักษณะตำหนิ → บนภาพจริงจะพัง.
  เทรนบน grayscale บังคับให้เรียนจากรูปร่าง/พื้นผิว

## 4. Class-balanced oversampling (แก้คลาสที่โมเดลมองข้าม)

จาก confusion matrix พบว่า crazing 64% และ rolled-in_scale 34% ถูกทำนายเป็น background
(missed detection ล้วน ไม่ใช่สับสนข้ามคลาส) สาเหตุ = class imbalance + mosaic ย่อ texture เต็มภาพ

```bash
python make_oversampled_list.py --dataset merged_dataset_gray   # หรือ merged_dataset ถ้าไม่ทำ grayscale
```

สร้าง `<dataset>/train_oversampled.txt` + `<dataset>/data_oversampled.yaml` (ทำซ้ำ path ในไฟล์ list ไม่ก็อปไฟล์จริง):

| คลาส | ตัวคูณ | images (หลัง) |
|---|---|---|
| crazing | ×3 | 720 |
| pitted_surface | ×2 | 482 |
| rolled-in_scale | ×2 | 480 |
| scratches | ×2 | 482 |
| ที่เหลือ | ×1 | เท่าเดิม |

ใช้คู่กับ `data_oversampled.yaml` (train ชี้ไป `train_oversampled.txt`)

## 5. เทรน

```bash
# baseline (augmentation เดิม, dataset สมดุลปกติ)
python train.py --data merged_dataset/data.yaml --name train-clean --epochs 100 --batch 8

# train-balanced เดิม (yolo11n): oversampling + texture-aware augmentation
python train.py --recipe texture --data data_oversampled.yaml --name train-balanced --epochs 100 --batch 8 --patience 30

# accuracy (Tier 1+2): label สะอาด + grayscale + yolo11s
python train.py --recipe texture --data merged_dataset_gray/data_oversampled.yaml \
                --model yolo11s.pt --name train-gray-s --epochs 120 --batch 6 --patience 40
```

recipe `texture`: `mosaic=0.3, close_mosaic=20, scale=0.2, degrees=5, flipud=0.5, erasing=0.2, cos_lr=True`
— ลด augmentation ที่ทำลาย texture ละเอียดเต็มภาพของ crazing / rolled-in_scale

## 5.5 per-class confidence threshold (accuracy — Tier 2)

pipeline ใช้ conf ค่าเดียวทุกคลาส แต่จุดทำงานที่ดีต่างกันมาก (rust มั่นใจสูงตลอด, crazing ยิงเบา)

```bash
python tune_thresholds.py --weights runs/detect/train-gray-s/weights/best.pt \
                          --data merged_dataset_gray/data.yaml
```

เขียน `thresholds.json` — `pipeline.py` / `app.py` / `evaluate_real.py` โหลดใช้อัตโนมัติถ้ามีไฟล์
(ปิดด้วย `--no-class-conf`)

## 6. ชุดทดสอบภาพจริง (สำหรับวัด pipeline end-to-end)

NEU/Rust/Crack เป็นภาพ crop แน่นอยู่แล้ว → วัด Stage 1 ไม่ได้
ต้องมีภาพเหล็กถ่ายจริงแยกต่างหาก วางที่:

```
real_test/
  images/          ภาพเหล็กถ่ายจริง 40–60 ภาพ (≥1000px ด้านสั้น)
  labels.csv       filename,classes   (เช่น  photo_001.jpg,rust;scratches  |  photo_002.jpg,none)
  SOURCES.md       ที่มาของแต่ละภาพ (ถ้าไม่ได้ถ่ายเอง)
```

วัดผลด้วย:

```bash
python evaluate_real.py --mode pipeline    # Stage 1 + Stage 2
python evaluate_real.py --mode baseline    # YOLO บนภาพเต็ม ไม่มี Stage 1 (ไว้เทียบ)
```
