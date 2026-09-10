# Steel Surface Defect Detection — ระบบตรวจจับตำหนิพื้นผิวเหล็ก (2-Stage)

ตรวจจับและจำแนกตำหนิบนพื้นผิวเหล็กจากภาพถ่าย โดยแบ่งเป็น 2 ขั้นตอน

| Stage | หน้าที่ | โมเดล |
|-------|---------|-------|
| **0. Steel Gate** (เดโมเท่านั้น, advisory) | จำแนก "เหล็ก / ไม่เหล็ก" — เตือนเมื่อภาพอาจไม่ใช่พื้นผิวเหล็ก | YOLO11n-cls (`steel_gate.py` / `train-gate`) |
| **1. Metal Localization** | หา region ในภาพที่เป็นวัสดุ "เหล็ก/โลหะ" แล้ว crop ออกมา | [DMS46](https://github.com/apple/ml-dms-dataset) (Apple Dense Material Segmentation, pre-trained, TorchScript) |
| **2. Defect Detection** | ตรวจชนิดตำหนิบนภาพที่ crop มา 8 ประเภท | YOLO11n (เทรนเองบน dataset รวม) |

```
ภาพถ่าย ──▶ [Stage 1: DMS46] ──▶ mask พื้นที่เหล็ก ──▶ crop
                                                        │
        ผลลัพธ์ ◀── วาดกรอบ + ป้ายไทย ◀── [Stage 2: YOLO11] ◀──┘
```

โปรเจคนี้ใช้คู่กับ repo `ml-dms-dataset` (โค้ดต้นฉบับของ Stage 1) — ไฟล์โมเดล `DMS46_v1.pt` ถูก copy มาไว้ใน repo นี้แล้ว

### 2 ราง โมเดล Stage 2 (แยกกันชัด)

| ราง | โมเดล | เทรนจาก | ใช้ที่ไหน |
|---|---|---|---|
| **เล่มจบ / benchmark** | `train-gray-n2` (grayscale) | NEU-DET + Roboflow (แล็บ) | ค่า default ของ `pipeline.py` / `evaluate.py` · ตัวเลขในภาคนิพนธ์ · `thresholds.json` |
| **เดโม / product** | `train-real3` (RGB) | + ภาพ corrosion ถ่ายจริง round 1+3 (~1,355 ภาพ) | ค่า default ในหน้า `app.py` · `thresholds_real3.json` |

> โมเดลเล่มจบ mAP50 บน benchmark สูง (0.867) แต่ **transfer ≈ 0 บนภาพถ่ายจริง** (domain gap) — โมเดลเดโม (`train-real3`) จับสนิมบนภาพจริงได้ (rust recall **12/12**, micro-F1 0.68) แลกกับ benchmark ที่ต่ำกว่าเล็กน้อย (mAP50 0.828) ดูหัวข้อ "ราง product / เดโม" และ "Pipeline end-to-end vs Baseline"

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
│
│  ── แกนระบบตรวจจับ ──
├── DMS46_v1.pt              โมเดล Stage 1 (TorchScript)
├── pipeline.py              รัน pipeline 2-stage เต็มระบบ  ← ไฟล์หลัก (build_regions / run_stage1 / run_stage2 / cross_region_nms)
├── steel_gate.py            Stage 0 — classifier เหล็ก/ไม่เหล็ก (advisory, เดโมเท่านั้น)
├── app.py                   Prototype UI (Gradio) — 3 แท็บ: อัปโหลด / ถ่ายภาพ / เรียลไทม์ (สตรีมกล้อง)
├── train.py                 เทรนโมเดล Stage 2  (--recipe {default,texture,camera,domainrand})
├── train_gate.py            เทรน Stage 0  (--recipe เดียว, yolo11n-cls)
├── run_round.py             รัน active-learning 1 รอบ (import → merge → train → eval)
├── test_smoke.py            smoke test — รัน pipeline 1 ภาพ + เช็คโครง output (กัน regression)
│
│  ── วัดผล ──
├── evaluate.py              วัดผลบน test split (Stage 2 mAP / pipeline image-level)
├── evaluate_real.py         วัดผลบนภาพเหล็กถ่ายจริง — pipeline vs baseline (--weights / --thresholds เลือกรางได้)
├── evaluate_stage1.py       วัด Stage 1 (DMS46) เชิงตัวเลข: detection rate / coverage / เวลา
├── evaluate_cross_dataset.py วัด generalization ข้ามชุด (GC10-DET)
├── tune_thresholds.py       หา per-class confidence จาก val → thresholds*.json
├── aggregate_seeds.py       รวมผล multi-seed → mean ± std
├── check_leakage.py         ตรวจ train/test leakage (perceptual hash + pixel cosine)
├── make_figures.py          สร้างรูปประกอบรายงานลง figures/
│
│  ── เตรียมข้อมูล ──
├── merge_datasets.py        รวม NEU + Rust + Crack เป็น 8 คลาส
├── resplit_grouped.py       group-aware stratified re-split (แก้ leakage — ตัวที่ใช้จริง)
├── resplit_dataset.py       re-split เดิม (ก่อนพบ leakage — เก็บไว้อ้างอิง)
├── fix_labels.py            รวมกล่อง crazing/rolled-in เป็น 1/ภาพ + ตัดกล่องเสีย
├── make_grayscale_dataset.py  merged_dataset/ → merged_dataset_gray/ (ตัด shortcut สี)
├── make_oversampled_list.py  class-balanced oversampling (--dataset) → <ds>/train_oversampled.txt
├── make_gate_dataset.py     สร้าง dataset เหล็ก/ไม่เหล็ก สำหรับ Stage 0
├── import_labeled.py        import ภาพ labeled จาก Roboflow/โฟลเดอร์ (active-learning)
├── annotate_bootstrap.py    ช่วย pre-annotate ภาพใหม่ด้วยโมเดลปัจจุบัน
│
│  ── config ──
├── data.yaml / data_oversampled.yaml   config dataset (6 คลาส NEU เดิม / 8 คลาส oversampled)
├── thresholds.json         per-class conf — ราง เล่มจบ (train-gray-n2)
├── thresholds_real3.json   per-class conf — ราง เดโม (train-real3, round 3) ← app.py default
├── thresholds_demo.json    per-class conf — train-real1 (round 1)
├── thresholds_real2.json   per-class conf — train-real2 (round 2)
│
│  ── เอกสาร / โน้ต ──
├── NEXT_STEPS.md            runbook + สถานะงาน + ผลการทดลองล่าสุด
├── thesis_notes.md          use case, Stage 1 = negative ablation, ผลรายรอบ
├── prepare_data.md          ขั้นตอนเตรียม dataset ตั้งแต่ raw source (reproducibility)
├── cross_dataset_eval.md / literature_comparison.md   ผล GC10 / เทียบเปเปอร์ NEU-DET
├── DATA_COLLECTION.md / SHOOTING_GUIDE.md   ราง product: เก็บภาพถ่ายจริงยังไง
├── make_thesis_doc.py + thesis_content.py   generate ภาคนิพนธ์ บท 1–3 (.docx)
├── make_ui_doc.py / make_uml_doc.py / make_design_doc.py / make_diagrams_doc.py /
│   make_database_doc.py / make_error_analysis_doc.py / make_combined_doc.py
│                            generate เอกสารประกอบเล่ม (.docx — ไม่เก็บใน git, รันสร้างใหม่ได้)
│
│  ── data / artifacts (gitignored — สร้างใหม่ได้) ──
├── merged_dataset/          8 คลาส RGB — labels_raw/ = label ก่อน fix_labels.py
├── merged_dataset_gray/     เวอร์ชัน grayscale (make_grayscale_dataset.py)
├── dataset_real/ dataset_gate/ downloads/   ข้อมูลราง product / Stage 0
├── train/ valid/ test/      NEU ดิบจาก Roboflow (source ของ merge_datasets.py)
├── rust_dataset/ crack_dataset/   dataset ดิบก่อน merge
├── real_test/               ภาพเหล็กถ่ายจริง 18 ภาพ + labels.csv (images/ tracked, ดู SOURCES.md)
├── test_images/             ภาพตัวอย่าง 8 คลาส + normal (ใช้ใน demo / smoke test — tracked)
├── external_test/gc10/      GC10-DET สำหรับ cross-dataset eval
├── runs/detect/             weights (เทรนใหม่ได้จาก train.py)
│   ├── train-gray-n2/  + -s{1,2,3}/   ราง เล่มจบ, yolo11n grayscale — multi-seed n=4  ← pipeline.py default
│   ├── train-gray-s2/                 yolo11s (model-size ablation)
│   ├── train-real3/                   ราง เดโม — + corrosion จริง round 1+3  ← app.py default
│   ├── train-real1/  train-real2/     ราง เดโม รุ่นก่อน (round 1 / round 2)
│   ├── train-dr/                      domain randomization (negative result)
│   ├── train-clean/  train-balanced/  baseline เดิม (ก่อนแก้ leakage)
│   └── train-gate/                    Stage 0
├── figures/ results/ pipeline_results/   รูป + ผล eval (JSON) — generate ได้เอง
└── weights/                yolo11*.pt ที่ดาวน์โหลดมา
```

ขั้นตอนเตรียมข้อมูลแบบละเอียด (raw source → merge → resplit → oversample) อยู่ใน **`prepare_data.md`**
สถานะงานล่าสุด + ผลการทดลองรายรอบ อยู่ใน **`NEXT_STEPS.md`**

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
python pipeline.py --image test_images/rust_example.jpg

# ทั้งโฟลเดอร์
python pipeline.py --folder test_images --output_dir pipeline_results

# ปรับ threshold / บังคับใช้ CPU / เลือกโมเดล Stage 2
python pipeline.py --image a.jpg --conf 0.35 --device cpu
python pipeline.py --image a.jpg --weights runs/detect/train-gray-n2/weights/best.pt
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
python app.py                 # http://127.0.0.1:7860 + วง LAN เดียวกัน
python app.py --share         # + ลิงก์สาธารณะ *.gradio.live (จำเป็นถ้าจะถ่ายจากกล้องมือถือ — ต้อง https)
python app.py --local-only    # เปิดเฉพาะเครื่องนี้
```

- **ไม่มีตัวเลือกให้ผู้ใช้ปรับ** — โมเดล (`train-real3`) + โหมด (มาตรฐาน) + threshold ตั้งค่าที่เหมาะสุดไว้แล้ว
  โค้ดยังสลับโมเดล/เปิด TTA-multiscale ได้ผ่าน `_STAGE2_MODELS` / State ใน `app.py`
- **3 แท็บรับภาพ:** อัปโหลด/วาง · ถ่ายจากกล้อง · **เรียลไทม์** (สตรีมกล้องต่อเนื่อง วาดกรอบ + นับตำหนิสดทุก ~0.3 วิ, ข้าม Stage 1) — แกลเลอรี `demo_samples/` 2 ภาพ/คลาส ขึ้นก่อน
- แสดง: การ์ดสรุปผล + ตารางรายการตำหนิ (เรียงตามความเสี่ยง) + **กล่องสาเหตุที่พบบ่อย + คำแนะนำ ต่อชนิดที่เจอ**
  (ข้อมูลอ้างอิงทั่วไป ไม่ใช่วินิจฉัยชิ้นงาน) + ภาพผลลัพธ์ · ดาวน์โหลดผล (zip) + เก็บ feedback ลง `demo_logs/`
- Stage 1 ยังทำงานเบื้องหลัง (fallback + cross-region NMS แบบเดียวกับ `pipeline.py`) แต่ไม่โชว์ภาพ metal region แล้ว

---

## ราง product / เดโม (แยกจากเล่มจบ)

โมเดลเล่มจบเทรนบน NEU-DET (แล็บ, close-up, grayscale) → **ไม่ transfer ไปภาพถ่ายจริง** ราง product
พยายามปิดช่องว่างนี้ (ผล + runbook เต็มใน `NEXT_STEPS.md` / `DATA_COLLECTION.md`):

| ความพยายาม | ผล |
|---|---|
| **`train-real1`** (round 1) — config เล่มจบ (RGB) + 672 ภาพ corrosion ถ่ายจริง | rust recall บนภาพจริง 0/12 → **8/12**, micro-F1 0.067 → **0.50** — default เดิม |
| **`train-real3`** (round 3) — + 683 ภาพ corrosion จริง (เสาส่งไฟ/ท่อ/แผ่นเหล็ก จาก `downloads/`) เทรนสด 100 epoch | ✅ **default ปัจจุบัน** — rust recall **8/12 → 12/12** (P 0.80 → 0.92), micro-F1 **0.50 → 0.68**, lab mAP50 0.828 · 6 คลาส texture ยังเท่าเดิม |
| **`train-real2`** (round 2) — + 1022 ภาพสนิม scene จริง | ⚠️ conf บนสนิม scene สูงขึ้น แต่ **regress บนสนิม lab-crop** — เป็น "รุ่นทดลอง" ไม่ใช่ default |
| **`train-dr`** — domain randomization (aug แรงสุด) | ❌ **negative result** — lab mAP50 ตกเหลือ 0.777, ไม่ช่วยภาพจริง (ไม่มีข้อมูล = aug สร้างความรู้ใหม่ไม่ได้) |
| **Stage 0 gate** (`steel_gate.py`) — classifier เหล็ก/ไม่เหล็ก | ⚠️ overfit ไป lab domain — ใช้แบบ **advisory** เท่านั้น (แปะหมายเหตุ ไม่ veto) |

**รากปัญหาที่เหลือ:** ไม่มีภาพถ่ายจริงของ 6 คลาส texture (crazing/inclusion/pitted/rolled-in/scratches + crack บนชิ้นงานจริง) — เดโมจึงใช้ได้จริงเฉพาะ **สนิม**

---

## ผลการทดลอง

> ### ⚠️ อัปเดต 2026-09-06 — Data leakage audit + retrain + multi-seed
>
> **1) Leakage:** `check_leakage.py` (perceptual hash + pixel cosine) พบชุด Roboflow
> "Danger-Rust" เป็นภาพถ่ายรัว → split เดิมสุ่มแยกเฟรมติดกันคนละ split →
> **rust ใน valid 100% / test 98% มีภาพเกือบเหมือนอยู่ใน train** (932 คู่)
> แก้ด้วย `resplit_grouped.py` (group-aware re-split, 80/10/10 → 3338/422/425, leakage = 0)
>
> **2) โมเดลหลักเปลี่ยนเป็น `train-gray-n2` (YOLO11n)** — เล็กกว่า 4x (5.5 MB), เร็วกว่า 2x (~5 ms),
> และ mAP สูงกว่า YOLO11s:
>
> | โมเดล (split สะอาด) | test mAP50 | mAP50-95 | หมายเหตุ |
> |---|---|---|---|
> | `train-gray-n2` (yolo11n) — **pipeline ใช้ตัวนี้** | **0.867 ± 0.010** | 0.536 ± 0.009 | multi-seed n=4 |
> | `train-gray-s2` (yolo11s) | 0.840 | 0.525 | single run (ต่ำกว่า n2 ~2.7σ) |
>
> **3) Multi-seed** (`aggregate_seeds.py`, n=4) — รายคลาส mAP50 (mean ± std):
>
> | class | mAP50 | recall | | class | mAP50 | recall |
> |---|---|---|---|---|---|---|
> | rust | 0.995 ± 0.00 | 1.00 | | pitted_surface | 0.803 ± 0.025 | 0.756 |
> | patches | 0.961 ± 0.011 | 0.895 | | rolled-in_scale | 0.819 ± 0.048 | 0.725 |
> | scratches | 0.947 ± 0.016 | 0.897 | | crazing | 0.867 ± 0.054 | 0.806 |
> | inclusion | 0.856 ± 0.016 | 0.789 | | **crack** | **0.687 ± 0.011** | **0.606** |
>
> **อ่านผล:** overall mAP50 0.853 (เดิม, leaky) → 0.867 ± 0.010 (n2, สะอาด) — ตัวเลขเดิมไม่ได้ผิดมาก
> และตอนนี้ป้องกันได้ · rust 0.995 คงที่ทุก seed แต่เพราะ subset rust เป็นกลุ่ม homogeneous
> แยกจาก NEU ง่าย (ดู `cross_dataset_eval.md` + error analysis EA-4 ใน `make_error_analysis_doc.py`) ไม่ใช่ leakage แล้ว ·
> **yolo11n > yolo11s → model size ไม่ใช่ปัจจัยหลัก** (เกนมาจาก label สะอาด + grayscale) ·
> crack อ่อนสุดและ crazing/rolled-in_scale std สูง (ไม่เสถียรข้าม seed)
>
> `thresholds.json` recompute สำหรับ n2: macro-F1 val 0.824 → 0.844
>
> หัวข้อย่อยด้านล่างที่อ้าง `train-gray-s` / `train-gray-n` (ไม่มี "2") เป็นตัวเลข **ก่อนแก้ leakage**
> เก็บไว้เทียบ before/after — ตัวเลข canonical ให้ยึดตารางนี้

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
(โมเดลหลักปัจจุบัน = `train-gray-n2` หลังแก้ leakage + multi-seed, `pipeline.py` ชี้ตัวนี้แล้ว — ดูกล่อง ⚠️ ด้านบน)

### ตารางเปรียบเทียบ

| Config | mAP50 | mAP50-95 | R | crazing mAP50 | rolled-in_scale mAP50 |
|---|---|---|---|---|---|
| train-2 (dataset ปนเปื้อน, test split เก่า) | 0.56 | – | – | – | – |
| train-clean (dataset สะอาด, aug default) | 0.750 | 0.448 | 0.703 | 0.423 | 0.585 |
| train-balanced (oversample + recipe texture) | 0.763 | 0.449 | 0.753 | 0.392 | 0.668 |
| train-gray-n (Tier 1+2, yolo11n — ablation) | 0.836 | 0.528 | 0.785 | 0.805 | 0.899 |
| **train-gray-s** (Tier 1+2, yolo11s) | **0.853** | **0.537** | **0.809** | **0.870** | **0.860** |

> ⚠️ **2 แถวล่างวัดคนละ protocol**: บน `merged_dataset_gray` test (ภาพ grayscale) และ label ของ
> crazing/rolled-in_scale ถูกรวมเป็น 1 กล่อง/ภาพ (`fix_labels.py`) — mAP ของ 2 คลาสนี้จึงวัด
> "หา region ตำหนิ" ไม่ใช่ "ระบุแต่ละ patch" เทียบตัวเลขตรง ๆ กับ 3 แถวบนไม่ได้
> (train-gray-n vs train-gray-s เทียบกันเองได้ — protocol เดียวกัน ดูหัวข้อ "Ablation — model size")

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

### Ablation — model size: `train-gray-n` (yolo11n) vs `train-gray-s` (yolo11s)

เทรน yolo11n บน `merged_dataset_gray` ด้วย recipe/oversampling ชุดเดียวกับ train-gray-s
(ต่างกันแค่ขนาดโมเดล) → แยกผลของ "label สะอาด + grayscale" ออกจาก "11n → 11s"
วัดบน `merged_dataset_gray` test (protocol เดียวกับ train-gray-s):

| Config | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| train-balanced (yolo11n, สี) วัดบน test เทา | 0.544 | 0.295 | 0.660 | 0.561 |
| **train-gray-n** (yolo11n + label สะอาด + gray) | **0.836** | 0.528 | 0.850 | 0.785 |
| **train-gray-s** (yolo11s + label สะอาด + gray) | **0.853** | 0.537 | 0.836 | 0.809 |

รายคลาส mAP50 (`figures/per_class_map.png`) — train-gray-n / train-gray-s แทบทับกันทุกคลาส
(gray-s นำ crazing/patches/pitted/scratches ~0.03–0.06, gray-n นำ rolled-in_scale/crack ~0.02–0.04)

**อ่านผล:** เกน Tier 1+2 เกือบทั้งหมด (0.544 → 0.836, **+0.29 mAP50**) มาจาก **label สะอาด + grayscale**
ส่วนการอัป yolo11n → yolo11s ได้เพิ่มแค่ **+0.017 mAP50** (recall +0.024) — model size ไม่ใช่ปัจจัยหลัก
ในโดเมนนี้ ถ้าต้องการ throughput สูง yolo11n คุ้มกว่า (2.6M param / 6.4 GFLOPs เทียบ yolo11s ~9.4M / 21.5 — เบากว่า ~3× ที่ mAP ตกแค่ ~2 จุด)

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

วัดบน `real_test/` **18 ภาพถ่ายจริงระดับ scene** (ดึงจากเน็ต — ข้อจำกัดของชุด ดู `real_test/SOURCES.md`:
เอียงไป rust, ไม่มี inclusion/rolled-in_scale/crazing) — image-level P/R/F1:

| โมเดล Stage 2 | | micro-P | micro-R | micro-F1 | macro-F1 | sec/img |
|---|---|---|---|---|---|---|
| `train-gray-n2` (เล่มจบ) | Pipeline (2-stage) | 0.143 | 0.043 | 0.067 | 0.057 | 0.56 |
| `train-gray-n2` (เล่มจบ) | Baseline (ไม่มี Stage 1) | 0.333 | 0.043 | 0.077 | 0.067 | 0.04 |
| `train-real1` (round 1) | Pipeline (2-stage) | 0.692 | 0.391 | 0.500 | 0.219 | 0.35 |
| `train-real1` (round 1) | Baseline (ไม่มี Stage 1) | 0.750 | 0.391 | 0.514 | 0.226 | 0.03 |
| **`train-real3`** (เดโม default) | Pipeline (2-stage) | **0.867** | **0.565** | **0.684** | 0.272 | 0.44 |
| **`train-real3`** (เดโม default) | Baseline (ไม่มี Stage 1) | **0.867** | **0.565** | **0.684** | 0.272 | 0.05 |

> วัดรุ่นเดโม: `python evaluate_real.py --weights runs/detect/train-real3/weights/best.pt --thresholds thresholds_real3.json`
> รายละเอียด round 3 (import ชุดไหน + คำสั่ง + ผลรายคลาส) อยู่ใน `NEXT_STEPS.md` หัวข้อ "Round 3 — train-real3"

**อ่านผล:**
- โมเดลเล่มจบ (`train-gray-n2`, grayscale + NEU benchmark) **แทบไม่ยิงบนภาพถ่ายจริงเลย** (micro-R 0.043,
  rust 0/12 เพราะ conf 0.92 จาก val แล็บกรองทิ้งหมด) — สอดคล้องกับ cross-dataset GC10-DET ที่ transfer ≈ 0
  → เป็นหลักฐาน **domain gap** ตรง ๆ ไม่ใช่จุดบกพร่องของสถาปัตยกรรม
- ราง เดโม: `train-real1` (+672 ภาพ corrosion round 1) → rust recall **0/12 → 8/12**, micro-F1 0.067 → 0.50 ;
  `train-real3` (+683 ภาพ corrosion round 3) → rust recall **8/12 → 12/12** (P 0.80 → 0.92), micro-F1 **→ 0.68**
  — ยืนยันซ้ำว่า **ยิ่งมีภาพในโดเมนเข้าชุดเทรน ยิ่งดีขึ้น** (ดู `DATA_COLLECTION.md` / `NEXT_STEPS.md`)
- **Stage 1 ไม่ช่วย** แม้บนภาพ scene จริง: `stage1_metal_found_rate` 0.56 (ดีกว่าบน crop แล็บ 0.35)
  แต่ pipeline micro-P **ต่ำกว่า** baseline ทั้งสองโมเดล (fallback + กรอบ metal เพี้ยนเพิ่ม FP)
  → conclusion เป็น **negative ablation** ตามที่ตั้งไว้ใน `thesis_notes.md` ข้อ 2

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

เทียบกับภาพถ่ายจริงระดับ scene (`real_test/`): DMS46 ตรวจเจอเหล็กบ่อยขึ้น (`metal_found_rate` 0.56)
แต่ยังตกไป fallback เป็นส่วนใหญ่ — ดูหัวข้อ "Pipeline end-to-end vs Baseline"

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
4. **Stage 1 (DMS46) ไม่ช่วยเชิงบวก — สรุปเป็น negative ablation** — บน `merged_dataset` test
   `evaluate_stage1.py` = fallback 78%; บน `real_test/` 18 ภาพ scene จริง Stage 1 ทำงานบ่อยขึ้น
   (metal_found_rate 0.56) แต่ pipeline micro-P **ต่ำกว่า** baseline ทั้งสองโมเดล
   → material segmentation ระดับฉากไม่ใช่ front-end ที่เหมาะกับโดเมนนี้; classifier เหล็ก/ไม่เหล็ก
   ตัวเล็กที่ fine-tune เองน่าจะเหมาะกว่า (future work)
5. **crack เป็นคลาสที่อ่อนที่สุด** — test mAP50 **0.687 ± 0.011**, recall **0.61** (n=4).
   สาเหตุ: (ก) มาจาก `crack_dataset` แหล่งเดียว annotate เป็น polygon → bbox หลวม
   (ข) เป็นตำหนิที่บางที่สุด grayscale ยิ่งลด contrast ของขอบรอย
   (ค) ภาพ crack ในโดเมนจริงมีสเกล/พื้นผิวหลากหลายกว่าคลาส NEU มาก
   สำหรับ prototype คัดกรอง recall 0.61 ยังใช้ได้ (คนตรวจซ้ำอยู่แล้ว) แต่เป็นจุดที่ต้องเพิ่มข้อมูลก่อนใช้จริง
6. **crazing ยังเป็นคลาส texture ที่ยากสุด** (std ข้าม seed ~0.05) — สอดคล้องกับ literature ของ NEU-DET
7. **domain gap รุนแรงบนภาพถ่ายจริง** — โมเดลเล่มจบ (`train-gray-n2`) บน `real_test/` micro-R 0.043,
   rust 0/12 — grayscale + benchmark แล็บทำให้ transfer ≈ 0 (ตรงกับ cross-dataset GC10)
   ต้องมีภาพในโดเมนเข้าชุดเทรน (ราง active-learning ใน `DATA_COLLECTION.md`) ถึงจะใช้งานภาพจริงได้

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
