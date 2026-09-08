"""
สร้างเอกสารเตรียมสอบปกป้องโปรเจค (Defense Prep) -> docs/defense_prep.docx

    python make_defense_prep_doc.py

เนื้อหาอิงโค้ดจริงในโปรเจค (pipeline.py / app.py / train.py / evaluate*.py) +
README.md + thesis_notes.md + results/*.json — จัดตามกรอบคำถามที่กรรมการสอบมักถาม
8 หัวข้อ (ภาพรวม / architecture / database / diagram / algorithm-AI / API-backend /
security / testing) + ตารางเวอร์ชัน AI ที่ใช้ + cheat sheet ตัวเลขที่ต้องจำ

ต้องมี: python-docx   (pip install python-docx)
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent
DOCX = BASE / "docs" / "defense_prep.docx"


# ============================================================ เนื้อหา

PITCH = (
    "ระบบตรวจจับตำหนิพื้นผิวเหล็กจากภาพถ่าย เป็น prototype “ผู้ช่วยคัดกรอง” "
    "(screening assistant) รันบนเครื่องเดียว ผู้ใช้เปิดหน้าเว็บ (Gradio) อัปโหลดหรือถ่ายภาพเหล็ก "
    "ระบบประมวลผล 2 ขั้นตอน: Stage 1 ใช้โมเดล material segmentation (DMS46 ของ Apple) "
    "หาบริเวณที่เป็นโลหะในภาพ แล้ว crop ออกมา; Stage 2 ใช้ YOLO11 ที่เทรนเอง ตรวจและจำแนก "
    "ตำหนิ 8 ชนิด (รอยแตกลายงา สิ่งแปลกปลอม ผิวลอก ผิวเป็นหลุม สะเก็ดรีด รอยขีดข่วน สนิม รอยแตกร้าว) "
    "บนบริเวณนั้น สุดท้ายวาดกรอบ + ป้ายภาษาไทย + จัดระดับความเสี่ยง + สรุปผลเป็นการ์ดและตาราง "
    "จุดเด่นคือทำงานได้แม้ Stage 1 หาโลหะไม่เจอ (fallback ตรวจทั้งภาพ) และตัดผลซ้ำข้ามบริเวณด้วย NMS"
)

PROBLEM = [
    ("ปัญหา", "การคัดกรองสภาพผิวเหล็กก่อนนำไปใช้/รับเข้าคลัง ทำด้วยสายตาคน ช้า ไม่สม่ำเสมอ "
     "และขึ้นกับประสบการณ์ผู้ตรวจ"),
    ("ระบบช่วยอะไร", "ช่วยชี้ว่าภาพนี้ “น่าจะมีตำหนิชนิดใด อยู่ตรงไหน” ให้คนตรวจโฟกัสจุดที่ควรดูซ้ำ "
     "ไม่ใช่ตัดสินใจแทนคน"),
    ("ผู้ใช้", "โกดัง / ร้านเหล็ก / โรงกลึง — พนักงานถ่ายรูปเหล็กเส้น/แผ่น/คอยล์ด้วยมือถือ ก่อนรับเข้า/ก่อนตัดใช้งาน"),
    ("ขอบเขต (สำคัญ — ต้องพูดเอง)", "เป็น prototype เชิงการศึกษา ไม่ใช่ระบบตรวจสอบติดตั้งใช้งานจริง "
     "ไม่รองรับสายพาน real-time ไม่ต่อ PLC ไม่มีบัญชีผู้ใช้"),
]

FLOW_END_TO_END = [
    "1. ผู้ใช้เปิดเบราว์เซอร์ที่ http://127.0.0.1:7860 (หน้า Gradio จาก app.py)",
    "2. อัปโหลด / วาง / ถ่ายภาพ  →  Gradio ส่ง numpy array ของภาพเข้าฟังก์ชัน analyze() ฝั่ง Python",
    "3. โหลดโมเดลครั้งแรกครั้งเดียว (Stage 0 gate, Stage 1 DMS46, Stage 2 YOLO ทั้ง 3 รุ่น) แล้ว cache ไว้",
    "4. Stage 0 (เดโม, advisory): classifier ประเมิน P(เป็นพื้นผิวเหล็ก) — ใช้เป็นหมายเหตุ ไม่ veto",
    "5. Stage 1: run_stage1() → DMS46 คืน label map → กรองเฉพาะคลาส “Metal” (index 22) → mask",
    "6. build_regions(): mask → contour → รวมกรอบที่ติดกัน → ถ้าพื้นที่โลหะ < 5% หรือไม่พบเลย เพิ่มกรอบ = ทั้งภาพ (fallback)",
    "7. Stage 2: วน crop แต่ละบริเวณ → run_stage2() → YOLO predict → กรองด้วย per-class confidence threshold",
    "8. cross_region_nms(): ตัด detection คลาสเดียวกันที่ทับกัน (IoU > 0.5) จากบริเวณที่ซ้อนกัน/ fallback",
    "9. แยกผลเป็น “ยืนยัน” (conf ≥ threshold ที่จูน) กับ “อาจมี” (ต่ำกว่าเกณฑ์แต่ยังพอมีสัญญาณ)",
    "10. วาดกรอบ + ป้ายไทย + สร้างการ์ดสรุป (verdict) + ตารางเรียงตามความเสี่ยง + ภาพ Stage 1 → ส่งกลับหน้าเว็บ",
    "11. ผู้ใช้ดาวน์โหลดผล (zip: ภาพ + JSON) และส่ง feedback ได้ (เก็บลง demo_logs/ บนเครื่อง)",
]

WHY_ARCH = [
    ("ทำไมแบ่ง 2 (จริง ๆ 3) ขั้นตอน แทน YOLO ตัวเดียว",
     "แนวคิด: ภาพถ่ายมือถือมีพื้นหลัง (พื้น/ผนัง/พาเลท) ที่ไม่ใช่เหล็ก — อยากให้ Stage 2 โฟกัสเฉพาะผิวเหล็ก "
     "ลด false positive จากพื้นหลัง. Stage 0 (gate) ไว้เตือนกรณีภาพไม่ใช่เหล็กเลย. "
     "หมายเหตุตามผลจริง: การทดลองบน real_test พบว่า Stage 1 ไม่ได้ช่วยเพิ่ม precision (ดูหัวข้อ Algorithm/AI ข้อ failure) "
     "จึงรายงานเป็น negative ablation — เก็บโค้ดไว้ แต่สรุปว่า 2-stage ไม่คุ้มสำหรับ use case นี้ ต้องพูดตรงนี้ให้ได้"),
    ("ทำไมใช้ DMS46 เป็น Stage 1",
     "DMS46 = Dense Material Segmentation ของ Apple (ECCV 2022) จำแนกวัสดุ 46 ชนิดระดับพิกเซล pre-trained มาแล้ว "
     "ไม่ต้องเทรนเอง เรียกใช้ผ่าน TorchScript. เลือกเพราะเป็นโมเดล material (ไม่ใช่ object) จึงตอบ “ตรงนี้เป็นโลหะไหม” ได้"),
    ("ทำไมใช้ YOLO11 เป็น Stage 2",
     "YOLO = one-stage object detector เร็ว เทรนง่าย มี ecosystem (Ultralytics) ครบ. เลือก YOLO11n (nano) "
     "เพราะ ablation แสดงว่ารุ่นใหญ่ (11s) ไม่ได้ดีกว่าอย่างมีนัยในโดเมนนี้ — 11n เบากว่า ~3 เท่า ที่ mAP ตกแค่ ~2 จุด"),
    ("ทำไมใช้ Gradio ทำ UI",
     "Gradio สร้างหน้าเว็บจากฟังก์ชัน Python ได้เลย ไม่ต้องเขียน frontend/JS แยก เหมาะกับ prototype ที่โฟกัสตัวโมเดล "
     "และเปิดให้เครื่องอื่นในวง LAN เข้าได้ทันที (หรือ --share ได้ลิงก์ชั่วคราว)"),
    ("ทำไมไม่มี backend แยก / ไม่มีฐานข้อมูล",
     "เป็น prototype เครื่องเดียว ประมวลผลทีละคำขอ ไม่มีผู้ใช้พร้อมกันหลายคน ไม่ต้องเก็บประวัติระยะยาว หรือ query ข้ามภาพ "
     "→ การใส่ REST API + DBMS จะเป็น over-engineering. ผลตรวจ persist เป็นไฟล์ JSON ต่อภาพ (write-once) พอแล้ว"),
    ("Frontend กับ backend คุยกันยังไง (ในบริบทนี้)",
     "ไม่มีการเรียก HTTP/REST ข้ามโปรเซส — Gradio รันเว็บเซิร์ฟเวอร์ในโปรเซส Python เดียวกับตัวโมเดล "
     "การกดปุ่มบนหน้าเว็บ = Gradio เรียกฟังก์ชัน Python (analyze) โดยตรงผ่าน event binding (.click / .change / .stream)"),
]

# ---------- Database ----------
DB_WHY = (
    "ระบบ**ไม่ใช้ DBMS** (ไม่มี RDBMS / NoSQL / ORM) และไม่มี server — เป็น prototype เครื่องเดียว "
    "ประมวลผลต่อคำขอ ไม่มีสถานะร่วมหลายผู้ใช้ ไม่ต้องการ transaction/ACID ข้ามคำขอ "
    "ข้อมูลทั้งหมด persist เป็นไฟล์บนดิสก์: ผลตรวจต่อภาพ = JSON (*_result.json), "
    "label ชุดเทรน = YOLO .txt, label ชุดทดสอบจริง = .csv, นิยามคลาส = ค่าคงที่ในโค้ด "
    "เหตุผลที่เลือก JSON แบบ nested: 1 คำขอ = 1 ไฟล์ผลลัพธ์พอดี ไม่ต้อง join ไม่มีอัปเดตย้อนหลัง (write-once)"
)
DB_ENTITIES = [
    ("RESULT", "ไฟล์ *_result.json ต่อภาพ", "image (path), metal_regions, metal_area_ratio, fallback_full_image, regions[]"),
    ("REGION", "embedded ใน RESULT.regions[]", "region_id (คีย์ในภาพนี้), box_xywh[4], detections[]"),
    ("DETECTION", "embedded ใน REGION.detections[]", "class (→DEFECT_CLASS), confidence, bbox_xywh, bbox_xyxy_crop, bbox_xyxy_global"),
    ("DEFECT_CLASS", "ค่าคงที่ในโค้ด (pipeline.DEFECT_CLASSES + DEFECT_INFO)", "id 0–7, name, name_th, risk"),
    ("DATASET_IMAGE / LABEL_BOX", "ไฟล์ภาพเทรน + <split>/labels/*.txt", "filename, split ; class_id, xc, yc, w, h (normalized YOLO)"),
    ("REAL_TEST_LABEL", "real_test/labels.csv", "filename, classes (คั่นด้วย ; — ระดับภาพ) สำหรับ evaluate_real.py"),
]
DB_RELATIONS = (
    "RESULT 1–N REGION ; REGION 1–N DETECTION ; DETECTION N–1 DEFECT_CLASS ; "
    "DATASET_IMAGE 1–N LABEL_BOX ; LABEL_BOX N–1 DEFECT_CLASS"
)
DB_IF_MIGRATE = [
    ("RESULT", "result_id PK, image_path, metal_regions, metal_area_ratio, fallback_full_image, created_at"),
    ("REGION", "region_id PK, result_id FK→RESULT, box_x, box_y, box_w, box_h"),
    ("DETECTION", "detection_id PK, region_id FK→REGION, class_id FK→DEFECT_CLASS, confidence, bbox_x1..y2 (global)"),
    ("DEFECT_CLASS", "class_id PK, name, name_th, risk"),
    ("DATASET_IMAGE / LABEL_BOX", "image_id PK ; label_id PK, image_id FK, class_id FK, xc, yc, w, h"),
    ("REAL_TEST_IMAGE / _LABEL_CLASS", "image_id PK ; (image_id FK, class_id FK) PK คู่ = junction table N–N"),
]
DB_QA = [
    ("PK / FK ในเอกสารนี้คืออะไร",
     "PK ปัจจุบัน = ตำแหน่งไฟล์ (path ของ *_result.json) และ region_id (ลำดับในภาพ); "
     "FK เชิงตรรกะ = DETECTION.class อ้าง DEFECT_CLASS.name. ถ้าย้ายเข้า RDBMS จะเป็น PK/FK จริงตามตารางข้อ 3.4"),
    ("ความสัมพันธ์ 1:1 / 1:N / N:M ตรงไหน",
     "1:N: RESULT→REGION, REGION→DETECTION, DATASET_IMAGE→LABEL_BOX. "
     "N:1: DETECTION→DEFECT_CLASS. "
     "N:M: ภาพ real_test ↔ คลาสตำหนิ (1 ภาพมีหลายคลาส, 1 คลาสอยู่หลายภาพ) — ปัจจุบันเก็บเป็น string คั่น ; "
     "ถ้า normalize ต้องแตกเป็น junction table"),
    ("ถ้าลบ RESULT (ไฟล์ผล) จะกระทบอะไร",
     "REGION/DETECTION อยู่ในไฟล์เดียวกัน หายไปด้วย (embedded) — ไม่มี orphan เพราะไม่มีตารางแยก. "
     "DEFECT_CLASS ไม่กระทบ (คงที่ในโค้ด). ถ้าเป็น RDBMS ควรตั้ง ON DELETE CASCADE จาก RESULT ลง REGION/DETECTION"),
    ("REAL_TEST_LABEL ผิด 1NF ไหม",
     "ผิด — เก็บหลายค่าใน field เดียว (classes = “rust;pitted_surface”) จงใจ เพื่อให้แก้ด้วย Excel/Notepad ง่าย "
     "ถ้าย้าย RDBMS ต้องแตกเป็น REAL_TEST_IMAGE + junction table"),
]

# ---------- Diagrams ----------
DIAGRAMS = [
    ("Use Case", "Actor 2 ราย. ผู้ใช้งาน: อัปโหลดภาพ, ปรับ confidence/โหมดตรวจ, ตรวจภาพ, ดูผล, บันทึก/ส่งออก. "
     "ผู้พัฒนา: เตรียม dataset, เทรน (train.py), วัดผล (evaluate*.py), ปรับ threshold, สร้างเอกสาร"),
    ("Context / DFD Level 0", "process 0 = ทั้งระบบ. รับภาพ+พารามิเตอร์จากผู้ใช้ผ่าน UI คืนภาพผล+รายการตำหนิ; "
     "External: ไฟล์โมเดล DMS46, library YOLO, ระบบไฟล์ท้องถิ่น. Roboflow เกี่ยวเฉพาะตอนเตรียมข้อมูล (offline)"),
    ("C4 Level 1 (System Context)", "ผู้ใช้ (เบราว์เซอร์/Gradio) และผู้พัฒนา (CLI) ↔ ระบบ ↔ ระบบภายนอก 3 ตัว "
     "(DMS46 TorchScript, Ultralytics YOLO runtime, file system). ไม่มีการเรียก API ผ่านเครือข่าย"),
    ("C4 Level 2 (Container)", "Web UI (Gradio), CLI, Pipeline Core (pipeline.py — orchestrate + cross-region NMS), "
     "Stage 1/2 Runtime, Model & Config Store (ไฟล์บนดิสก์), Training/Eval Scripts. ไม่มี container ฐานข้อมูล"),
    ("ERD (Conceptual)", "ไม่มี DB จริง. " + DB_RELATIONS + " — สัญกรณ์ตีนกา (crow's foot)"),
    ("Class Diagram", "โค้ด functional — โมดูล pipeline/app/evaluate* เป็น «utility» (static operation + ค่าคงที่). "
     "«data» Detection / RegionMeta / Summary = dict ที่ pipeline สร้างและส่งต่อ (Summary = โครงของ *_result.json)"),
    ("Sequence — analyze()", "ผู้ใช้กด “ตรวจสอบ” → (ครั้งแรก) _ensure_models → _to_bgr → run_stage1 → build_regions "
     "→ loop run_stage2 ต่อ region (แปลง gray + กรอง threshold ในฟังก์ชัน) → cross_region_nms → วาดผล + verdict → return"),
    ("Activity — process_image()", "อ่านภาพ → run_stage1 → build_regions (จุดตัดสินใจ metal_ratio < 0.05 → เพิ่มกรอบทั้งภาพ) "
     "→ Pass 1: loop crop + run_stage2 + แปลง bbox เป็นพิกัดภาพเต็ม → cross_region_nms → Pass 2: วาดผล/ทำเครื่องหมายปกติ → เขียน .jpg + .json"),
    ("State Machine (ภาพ 1 รูป)", "Loaded → Stage1Done → RegionsReady (+fallback ถ้า ratio<0.05) → Stage2Done → "
     "Filtered (หลัง NMS+threshold) → Rendered. อยู่ใน local variable ไม่ persist"),
    ("Deployment", "โน้ตบุ๊ก Windows 11 เครื่องเดียว: Python 3.11 venv (PyTorch 2.6 / Ultralytics 8.4.126 + DMS46_v1.pt "
     "+ best.pt + thresholds.json), เบราว์เซอร์ต่อ Gradio ที่ 127.0.0.1:7860, GPU RTX 3050 ผ่าน CUDA 12.4. ไม่มี server/cloud"),
]

# ---------- Algorithm / AI ----------
AI_STAGES = [
    ("Stage 0 — Steel Gate (เดโม, advisory)",
     "Input: ภาพ BGR. Process: YOLO11n-cls จำแนก “เหล็ก / ไม่เหล็ก” คืนความน่าจะเป็น P(เหล็ก). "
     "Output: ตัวเลข 0–1 ใช้เป็นหมายเหตุเท่านั้น (classifier ยัง bias ไปภาพแล็บ จึงไม่ให้ veto). "
     "ถ้า P(เหล็ก) ต่ำมาก + Stage 1 ไม่พบโลหะ + Stage 2 เงียบ → แปะข้อความ “อาจไม่ใช่พื้นผิวเหล็ก”"),
    ("Stage 1 — Metal Localization (DMS46)",
     "Input: ภาพ BGR. Process: resize ด้านยาว = 512 รักษาสัดส่วน → normalize แบบ ImageNet (สเกล 0–255) → "
     "DMS46 (TorchScript) คืน label map [H,W] (argmax ในตัวโมเดลแล้ว) → เก็บเฉพาะพิกเซลที่ = คลาส Metal (output index 22) "
     "→ resize mask กลับเท่าภาพเดิม. Output: mask ไบนารี (255 = โลหะ)"),
    ("build_regions() — mask → กรอบ + fallback",
     "morphology open/close ล้าง noise → findContours → ตัดกรอบจิ๋ว/กรอบที่ fill ต่ำ → _merge_close_boxes รวมกรอบที่ห่างกัน "
     "≤ 3% ของด้านสั้น. ถ้า metal_ratio < 0.05 หรือไม่พบกรอบเลย → เพิ่มกรอบ (0,0,W,H) = ทั้งภาพ (fallback). "
     "คืน boxes + meta {metal_ratio, fallback_full_image, n_regions}"),
    ("Stage 2 — Defect Detection (YOLO11)",
     "Input: crop ของแต่ละบริเวณ. Process: ถ้าโมเดลเทรนบน grayscale จะแปลง crop เป็น grayscale ก่อน (กัน train/serve skew) → "
     "YOLO.predict ที่ conf ต่ำสุดก่อน → กรองแต่ละกล่องด้วย per-class threshold (thresholds.json). "
     "Output: list ของ {class, confidence, bbox} เรียงตาม confidence"),
    ("รวมผล — cross_region_nms()",
     "รวม detection ทุกบริเวณเป็นพิกัดภาพเต็ม (bbox_xyxy_global) → class-aware NMS: ถ้าคลาสเดียวกัน + IoU > 0.5 กับกล่องที่ conf สูงกว่า → ตัดทิ้ง "
     "(กันผลซ้ำจากกรอบ metal ที่ทับกัน หรือ fallback ทั้งภาพซ้อนกับกรอบ metal)"),
]

AI_DATA = [
    ("แหล่งข้อมูล (3 แหล่ง รวมเป็น 8 คลาส)",
     "NEU-DET 6 คลาส (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches — ภาพแล็บ grayscale ระยะใกล้) "
     "+ Rust dataset (Roboflow: RUST + DANGER-RUST) + Crack dataset (Roboflow, label เป็น polygon → แปลงเป็น bbox)"),
    ("การเตรียม (reproducible, seed=0)",
     "merge_datasets.py (กรอง NEU ให้สะอาด ~1439 ไฟล์) → resplit_grouped.py (group-aware stratified 80/10/10 = 3338/422/425 "
     "ทุก split ครบ 8 คลาส) → fix_labels.py (รวมกล่อง crazing/rolled-in เป็น 1/ภาพ) → make_grayscale_dataset.py (ตัด color shortcut) "
     "→ make_oversampled_list.py (class-balanced oversampling: crazing ×3, pitted/rolled-in/scratches ×2)"),
    ("Data leakage ที่เจอและแก้",
     "check_leakage.py (perceptual hash + pixel cosine) พบชุด Roboflow “Danger-Rust” เป็นภาพถ่ายรัว (burst) → "
     "split เดิมสุ่มแยกเฟรมติดกันคนละ split → rust ใน valid 100% / test 98% เกือบเหมือน train (932 คู่) → "
     "แก้ด้วย resplit_grouped.py (จับ near-duplicate ไว้ split เดียวกัน) → leakage = 0"),
    ("แบ่ง Train / Validation / Test",
     "Train 3338 / Valid 422 / Test 425 (≈ 80/10/10). Train ใช้ oversampled list. "
     "Validation ใช้ตอนเทรน (early stopping) + ใช้จูน per-class threshold. Test ใช้รายงานผลสุดท้าย (ไม่เคยเห็นตอนเทรน/จูน)"),
]

AI_MODEL_CHOICE = [
    ("ทำไม YOLO11 (ไม่ใช่ Faster R-CNN / RT-DETR)",
     "use case = คัดกรอง ไม่ต้องวัดขนาดตำหนิละเอียด, one-stage เร็วกว่า, Ultralytics ecosystem ครบ (train/val/export). "
     "literature: YOLO บน NEU-DET ปกติ mAP ~0.73–0.79 — เป็นฐานที่สมเหตุสมผล"),
    ("ทำไม YOLO11n (nano) ไม่ใช่ 11s",
     "ablation บน split สะอาด: train-gray-n2 (11n) mAP50 0.867 ± 0.010 > train-gray-s2 (11s) 0.840. "
     "11n มี 2.6M param / 6.4 GFLOPs, 11s ~9.4M / 21.5 — เบากว่า ~3 เท่า ที่ mAP สูงกว่าเล็กน้อย → model size ไม่ใช่ปัจจัยหลัก"),
    ("ทำไมแปลงเป็น grayscale",
     "rust/crack เดิมเป็นภาพสี ส่วน NEU เป็น grayscale → โมเดลอาจแยกคลาสจาก “โทนสี” แทนลักษณะตำหนิ (color shortcut). "
     "หลักฐาน: โมเดลที่เทรนบนภาพสี พอวัดบน test ที่เป็น grayscale → rust mAP 0.995 → 0.19 (recall = 0). "
     "จึงเทรน+วัดบน grayscale ทั้งหมดเพื่อบังคับให้เรียนจาก texture"),
    ("recipe augmentation (train.py --recipe)",
     "default = mosaic เต็ม + scale/degrees สูง (ดีกับตำหนิเป็นก้อนชัด). "
     "texture (ใช้กับเล่มจบ) = ลด mosaic เป็น 0.3, close_mosaic เร็ว, scale 0.2 — รักษา texture ที่กินทั้งภาพ (crazing = ร่างแหรอยแตก). "
     "camera = จำลองภาพถ่ายมือถือ (แสง/มุม/เบลอ) ใช้กับราง product. "
     "domainrand = สุ่ม appearance สุดขั้ว — ทดลองแล้ว negative result"),
]

METRICS_DEF = [
    ("Precision", "ของที่ระบบบอกว่า “เจอ” มีจริงกี่ %  =  TP / (TP + FP)  — สูง = เตือนผิดน้อย"),
    ("Recall", "ตำหนิที่มีจริงทั้งหมด ระบบจับได้กี่ %  =  TP / (TP + FN)  — สูง = พลาดน้อย (สำคัญสุดใน use case คัดกรอง)"),
    ("F1", "ค่าเฉลี่ยฮาร์มอนิกของ precision กับ recall  =  2PR / (P + R)  — สมดุลทั้งสองด้าน"),
    ("mAP@0.5 (mAP50)", "average precision เฉลี่ยทุกคลาส โดยนับว่ากล่องถูกเมื่อ IoU กับ ground truth ≥ 0.5"),
    ("mAP@0.5:0.95", "เฉลี่ย mAP ที่ IoU 0.5, 0.55, ..., 0.95 — เข้มกว่า วัดความแม่นของตำแหน่งกรอบด้วย"),
    ("Confusion Matrix", "ตารางแถว = คลาสจริง, คอลัมน์ = คลาสที่ทำนาย. แนวทแยง = ทำนายถูก. "
     "ในโปรเจคนี้: คลาสที่อ่อน (crazing/rolled-in) หลุดเป็น “background” ไม่ใช่สับสนข้ามคลาส → ปัญหาคือ recall ต่ำ"),
]

RESULTS_PERCLASS = [
    ("rust", "0.995", "สูงสุด — แต่ subset เป็นกลุ่ม homogeneous แยกง่าย"),
    ("patches", "0.961", ""),
    ("scratches", "0.947", ""),
    ("crazing", "0.867", "std ข้าม seed สูง (~0.05) — ไม่เสถียร"),
    ("inclusion", "0.856", ""),
    ("rolled-in_scale", "0.819", "std สูง (~0.05)"),
    ("pitted_surface", "0.803", ""),
    ("crack", "0.687", "อ่อนสุด — polygon→bbox หลวม, ตำหนิบางสุด, grayscale ลด contrast ขอบรอย"),
]

AI_FAILURE = [
    ("Stage 1 (DMS46) ไม่ช่วยเชิงบวก — negative ablation",
     "บน test crop แล็บ 416 ภาพ: fallback_rate 78%, gt_area_kept 16% (ถ้าไม่ fallback จะตัดตำหนิจริงทิ้ง ~84%). "
     "บน real_test 18 ภาพ scene จริง Stage 1 ทำงานบ่อยขึ้น (metal_found_rate 0.56) แต่ pipeline micro-precision ต่ำกว่า baseline "
     "(ไม่มี Stage 1) ทั้งสองรุ่นโมเดล → กรอบ metal ที่เพี้ยน + fallback ทำให้เกิด FP เพิ่มโดยไม่ได้ recall กลับมา. "
     "สรุปเล่ม: material segmentation ระดับฉากไม่เหมาะเป็น front-end ของโดเมนนี้ (future work = classifier เหล็ก/ไม่เหล็กตัวเล็ก fine-tune เอง)"),
    ("โมเดลเล่มจบ transfer ≈ 0 บนภาพถ่ายจริง (domain gap)",
     "train-gray-n2 บน real_test: micro-recall 0.043, rust 0/12 (per-class conf 0.92 ที่จูนจาก val แล็บ กรองทิ้งหมด). "
     "ตรงกับผล cross-dataset บน GC10-DET ที่ transfer ~0 → เป็นหลักฐาน domain gap ตรง ๆ ไม่ใช่จุดบกพร่องสถาปัตยกรรม"),
    ("ราง product (train-real1) กู้เฉพาะสนิม",
     "config เดียวกับเล่มจบ + 672 ภาพ corrosion ถ่ายจริง (RGB, recipe camera): rust recall บนภาพจริง 0/12 → 8/12, "
     "micro-F1 0.067 → 0.50. แต่ 5 คลาส texture อื่น ยัง 0 — ไม่มีข้อมูลถ่ายจริงของคลาสเหล่านั้น"),
    ("Domain randomization (train-dr) — negative result",
     "augmentation แรงสุด (photometric/scale/perspective สุดขั้ว + copy_paste + randaugment) → lab mAP50 ตกเหลือ 0.777 "
     "และไม่ช่วยภาพจริง. ยืนยันว่า augmentation ยืดได้แค่สิ่งที่มีในข้อมูล สร้างความรู้ใหม่ไม่ได้"),
    ("crack เป็นคลาสที่อ่อนสุด",
     "test mAP50 0.687 ± 0.011, recall 0.61. สาเหตุ: มาจากแหล่งเดียว label polygon→bbox หลวม, ตำหนิบางสุด, grayscale ลด contrast"),
    ("ถ้า AI ทายผิด เกิดจากอะไร (ตอบรวม)",
     "(1) domain gap — เทรนจากภาพแล็บ close-up ทดสอบภาพถ่ายมือถือ scene. "
     "(2) class imbalance — คลาส texture instance น้อยกว่า crack/rust. "
     "(3) mosaic augmentation ย่อ texture จนหาย → โมเดลตอบ background. "
     "(4) label noise — crazing/rolled-in เดิม annotate เป็นกล่องย่อยมั่ว, crack เป็น polygon แปลง"),
]

# ---------- API / Backend ----------
API_POINTS = [
    ("ไม่มี REST API / ไม่มี HTTP method (GET/POST/...) ในความหมายปกติ",
     "Gradio รันเว็บเซิร์ฟเวอร์ในโปรเซสเดียวกับโมเดล. หน้าเว็บ (client) คุยกับเซิร์ฟเวอร์ Gradio ผ่าน HTTP/WebSocket ภายใน "
     "แต่โค้ดเราไม่ได้นิยาม endpoint เอง — เราผูก event เข้ากับฟังก์ชัน Python"),
    ("การผูก event (แทน routing)",
     "btn.click(analyze, inputs=[...], outputs=[...]) — กดปุ่ม “ตรวจสอบ” เรียก analyze(). "
     "inp.change(...) — อัปโหลดภาพเสร็จเรียกอัตโนมัติ. rt_in.stream(analyze_stream, ...) — โหมดเรียลไทม์ สตรีมเฟรมกล้อง. "
     "sens/model_sel/conf .change(...) — เปลี่ยนตัวเลือกแล้วตรวจภาพล่าสุดซ้ำ"),
    ("ตอนกด “ตรวจสอบ” เกิดอะไรขึ้น (ลำดับ)",
     "เบราว์เซอร์ส่งภาพ (numpy) + พารามิเตอร์ → Gradio เรียก analyze() → (ครั้งแรก) โหลด+cache โมเดล → "
     "Stage 0 gate → Stage 1 (run_stage1 + build_regions) → Stage 2 loop (run_stage2 + per-class threshold) → "
     "cross_region_nms → แยก confirmed/tentative → วาดภาพ + สร้าง verdict + ตาราง → return กลับหน้าเว็บ → "
     "แล้วเรียก prepare_download() ทำ zip ให้ดาวน์โหลด"),
    ("ทำไมไม่ให้ frontend เข้าถึงโมเดล/ไฟล์โดยตรง",
     "เบราว์เซอร์รันโค้ดไม่ได้แตะ Python/ไฟล์เครื่องอยู่แล้ว — Gradio เป็นชั้นกลางที่รับ input, เรียกฟังก์ชัน, ส่ง output กลับ "
     "โมเดลและไฟล์ทั้งหมดอยู่ฝั่งเซิร์ฟเวอร์ (โปรเซส Python)"),
    ("โหมด --share",
     "Gradio เปิด tunnel ให้ลิงก์ *.gradio.live (อยู่ ~72 ชม.) ไว้ให้คนนอกวง LAN ทดลอง — "
     "ภาพจะผ่านเซิร์ฟเวอร์ relay ของ Gradio ด้วย (ข้อควรระวังด้าน privacy — ดูหัวข้อ Security)"),
]

# ---------- Security ----------
SEC_POINTS = [
    ("ไม่มีบัญชีผู้ใช้ / ไม่มี authentication / authorization",
     "เป็น prototype รันในเครื่อง/วง LAN ที่เชื่อถือได้ ไม่มีข้อมูลผู้ใช้ให้ป้องกัน ไม่มีสิทธิ์ให้แบ่ง — "
     "จึงไม่มี login, JWT, session, RBAC. ถ้าจะ deploy จริงต้องเพิ่มชั้นนี้"),
    ("ไม่มี password / credential / API key ในโค้ด",
     "ระบบไม่ต่อบริการภายนอกที่ต้องยืนยันตัวตน (โมเดลเป็นไฟล์ local, dataset ดาวน์โหลดมาก่อนแล้ว) — ไม่มี secret ให้รั่ว"),
    ("Privacy — ประมวลผลบนเครื่อง",
     "ค่าเริ่มต้น: ภาพไม่ออกนอกเครื่อง/วง LAN. feedback + ภาพที่ผู้ใช้ส่ง เก็บใน demo_logs/ บนเครื่องนั้นเท่านั้น. "
     "ยกเว้นเปิด --share → ภาพผ่าน relay ของ Gradio (ต้องแจ้งผู้ใช้)"),
    ("Input validation ที่มี",
     "รับเฉพาะสกุลภาพที่กำหนด (IMAGE_EXTS). แปลง grayscale/RGBA เป็น BGR. guard: ภาพเปิดไม่ได้ → คืน error card, "
     "crop เล็กกว่า 8 px → ข้าม. analyze() ครอบ try/except ทั้งก้อน — error แสดงเป็นการ์ด ไม่ทำหน้าเว็บค้าง"),
    ("SQL Injection / injection อื่น",
     "ไม่มี SQL (ไม่มี DB) และไม่รับ query/path จากผู้ใช้ไปต่อคำสั่ง — ไม่มีช่องนี้. "
     "ไฟล์ผลลัพธ์ตั้งชื่อจาก stem ของไฟล์ input ฝั่งเซิร์ฟเวอร์เท่านั้น"),
    ("“ถ้ามีคนรู้ URL ของเรา เข้าระบบได้เลยไหม”",
     "ในวง LAN เดียวกัน = เข้าได้ (ไม่มี auth) — โดยตั้งใจ เพราะเป็น prototype ให้เพื่อน/อาจารย์ลองในวงเดียวกัน. "
     "ถ้า --share ใครมีลิงก์ก็เข้าได้จนกว่าจะปิด. ทางแก้ถ้าจะจริงจัง: auth (เช่น Gradio auth=(user,pass)) หรือวางหลัง reverse proxy"),
]

# ---------- Testing ----------
TEST_LEVELS = [
    ("Smoke / regression", "test_smoke.py", "รัน pipeline 1 ภาพจริง + เช็ค helper (build_regions, cross_region_nms, load_class_conf) "
     "+ เช็คโครง output — กันของพังหลังแก้โค้ด"),
    ("Component — Stage 2", "evaluate.py --mode stage2", "mAP50 / mAP50-95 / P / R ต่อคลาส บน test split"),
    ("Component — Stage 1", "evaluate_stage1.py", "detection rate / coverage / gt_area_kept / latency ของ DMS46 บน 416 ภาพ"),
    ("System — ภาพถ่ายจริง", "evaluate_real.py", "image-level P/R/F1 บน real_test/ 18 ภาพ + เทียบ pipeline vs baseline (ไม่มี Stage 1)"),
    ("Generalization", "evaluate_cross_dataset.py", "เทรน NEU-style → ทดสอบ GC10-DET (คนละชุด) วัด transfer"),
    ("Robustness (สถิติ)", "aggregate_seeds.py (n=4)", "เทรน 4 seed แล้วรายงาน mean ± std — กันผลฟลุ๊คจาก seed เดียว"),
    ("Data integrity", "check_leakage.py", "ตรวจภาพ near-duplicate ข้าม train/val/test (perceptual hash + pixel cosine)"),
]
TEST_CASES = [
    ("อัปโหลดภาพ rust (แล็บ)", "rust_example.jpg", "การ์ด “พบตำหนิความเสี่ยงสูง — สนิม” + กรอบแดง", "ผ่าน (conf 0.42)"),
    ("อัปโหลดภาพเหล็กสะอาด", "normal_steel_example.jpg", "การ์ด “ไม่พบตำหนิพื้นผิว”", "ผ่าน"),
    ("อัปโหลดภาพไม่ใช่เหล็ก", "ภาพเอกสาร/สัตว์", "แปะหมายเหตุ “อาจไม่ใช่พื้นผิวเหล็ก” (ไม่ฟันธง)", "ผ่านบางส่วน (gate ยัง bias)"),
    ("ไฟล์ผิดสกุล", ".txt / .pdf", "Gradio ปฏิเสธที่ตัวเลือกไฟล์ / คืน error card", "ผ่าน"),
    ("โหมดความไว “ไวมาก” บนผนังปูน", "real_015.jpg", "แสดงเป็น “อาจมี” เท่านั้น ไม่ขึ้น “ความเสี่ยงสูง”", "ผ่าน (หลังแก้ 2026-09-08)"),
    ("Stage 1 หาโลหะไม่เจอ", "ภาพสนิมเต็มเฟรม", "fallback ตรวจทั้งภาพ ไม่คืน “ไม่พบเหล็ก”", "ผ่าน"),
]

# ---------- Key AI versions ----------
AI_VERSIONS = [
    ("ภาษา / รันไทม์", "Python", "3.11", "venv บน Windows 11"),
    ("Deep learning framework", "PyTorch", "2.6.0+cu124", "torchvision 0.21.0+cu124"),
    ("GPU / CUDA", "NVIDIA RTX 3050 (6 GB)", "CUDA 12.4", "CPU รันได้แต่ช้า"),
    ("Stage 2 detector (library)", "Ultralytics", "8.4.126", "API เทรน/วัด/export"),
    ("Stage 2 — สถาปัตยกรรมโมเดล", "YOLO11n (nano)", "—", "2.6M param / 6.4 GFLOPs ; เทรนเอง 8 คลาส"),
    ("Stage 2 — น้ำหนักเริ่มต้น", "yolo11n.pt (COCO pre-trained)", "—", "ดาวน์โหลดจาก Ultralytics ครั้งแรก"),
    ("Stage 1 — โมเดล", "DMS46 (Apple ml-dms-dataset)", "v1 (DMS46_v1.pt)", "TorchScript, pre-trained, ไม่ fine-tune ; ECCV 2022"),
    ("Stage 0 — โมเดล (เดโม)", "YOLO11n-cls", "—", "classifier เหล็ก/ไม่เหล็ก, advisory"),
    ("UI", "Gradio", "6.26 (requirements: >= 4.44)", "เว็บจากฟังก์ชัน Python"),
    ("ประมวลผลภาพ", "OpenCV (opencv-python)", "5.0 (requirements: >= 4.9)", "อ่าน/แปลงสี/วาดกรอบ"),
    ("Array / รูป / เอกสาร", "NumPy / Pillow / python-docx / matplotlib", "2.4 / 10+ / 1.2 / 3.8+", "ประมวลผล + สร้างรูป/เอกสาร"),
]

MODELS_TRACK = [
    ("train-gray-n2", "เล่มจบ / benchmark", "YOLO11n, grayscale, NEU+Roboflow (แล็บ)",
     "test mAP50 0.867 ± 0.010 (n=4) — ค่า default ของ pipeline.py"),
    ("train-gray-s2", "ablation ขนาดโมเดล", "YOLO11s, grayscale", "test mAP50 0.840 — ใหญ่กว่า 3–4× แต่ไม่ดีกว่า"),
    ("train-real1", "เดโม / product (default ใน app.py)", "config เดียวกัน (RGB) + 672 ภาพ corrosion จริง, recipe camera",
     "rust recall บนภาพจริง 0/12 → 8/12 ; micro-F1 0.067 → 0.50"),
    ("train-real2", "รุ่นทดลอง (round 2)", "+ 1022 ภาพสนิม scene จริง", "conf บนสนิม scene สูงขึ้น แต่ regress บนสนิม lab-crop"),
    ("train-dr", "ทดลอง (negative result)", "domain randomization — aug แรงสุด", "lab mAP50 ตกเหลือ 0.777 ไม่ช่วยภาพจริง"),
    ("train-clean / train-balanced", "baseline เดิม (ก่อนแก้ leakage)", "8 คลาส RGB, aug default / + oversampling",
     "0.750 / 0.763 — เก็บไว้เทียบ ablation B"),
]

CHEAT = [
    "โมเดลหลัก (เล่มจบ): YOLO11n grayscale ชื่อ run = train-gray-n2 ; test mAP50 = 0.867 ± 0.010 (multi-seed n=4)",
    "mAP50-95 = 0.536 ± 0.009 ; precision 0.853 ; recall 0.809",
    "คลาสอ่อนสุด = crack (mAP50 0.687, recall 0.61) ; คลาสแข็งสุด = rust (0.995)",
    "8 คลาส: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches (จาก NEU) + rust + crack (Roboflow)",
    "split: 3338 / 422 / 425 (train/valid/test ≈ 80/10/10) หลังแก้ leakage (group-aware re-split)",
    "Stage 1 = DMS46, คลาส Metal = output index 22, input resize ด้านยาว 512 + ImageNet norm สเกล 0–255",
    "fallback: metal_ratio < 0.05 → ตรวจทั้งภาพ ; cross-region NMS IoU > 0.5 ; per-class conf จาก thresholds.json",
    "Stage 1 latency ~208 ms/ภาพ (GPU) ; Stage 2 ~10 ms/region",
    "ผลสำคัญ 3 ข้อ: (1) Stage 1 ไม่ช่วย = negative ablation (2) model size ไม่ใช่ปัจจัยหลัก (11n > 11s) "
    "(3) domain gap รุนแรง — โมเดลเล่มจบ transfer ≈ 0 บนภาพถ่ายจริง",
    "libraries: PyTorch 2.6.0+cu124, Ultralytics 8.4.126, Gradio 6.26, OpenCV 5.0, Python 3.11",
    "literature: YOLO บน NEU-DET ปกติ mAP ~0.73–0.79 → ผลเรา 0.867 อยู่ช่วงบน (แต่เทียบตรงไม่ได้ เพราะ 8 คลาส + grayscale + รวม 3 แหล่ง)",
]

SEVEN_Q = [
    "1. ทำอะไร?  2. ใครใช้?  3. Input คืออะไร?  4. Process ทำงานอย่างไร?",
    "5. Output คืออะไร?  6. ข้อมูลเก็บที่ไหน?  7. ทำไมถึงออกแบบแบบนี้?",
]

WHY_DRILL = [
    ("ทำไมเลือก Python", "ecosystem AI/CV ครบสุด (PyTorch, Ultralytics, OpenCV) prototype เร็ว"),
    ("ทำไมเลือก Gradio ไม่ทำ Flutter/React", "โปรเจคเน้นตัวโมเดล ไม่ใช่ UI — Gradio สร้างเว็บจากฟังก์ชัน Python ได้ทันที "
     "ไม่ต้องแยก frontend/backend/API ให้เสียเวลา และเปิดวง LAN / --share ได้เลย"),
    ("ทำไมไม่ใช้ฐานข้อมูล (PostgreSQL/MySQL)", "prototype เครื่องเดียว 1 คำขอ = 1 ผลลัพธ์ write-once ไม่ query ข้ามภาพ "
     "ไม่มีผู้ใช้พร้อมกัน → RDBMS เป็น over-engineering ; เก็บ JSON ต่อภาพพอ (มีแนวทาง schema ถ้าต้องย้ายในเอกสาร DB)"),
    ("ทำไม 2-stage ไม่ใช่ YOLO ตัวเดียว", "แนวคิดคือกันพื้นหลังที่ไม่ใช่เหล็ก — แต่ผลจริงบอกว่าไม่คุ้ม (negative ablation) "
     "ต้องพูดตรง ๆ ว่าเราทดลองแล้วและรายงานผลนี้ เป็น contribution ที่ให้เหตุผลกับสถาปัตยกรรมที่ง่ายกว่า"),
    ("ทำไม grayscale", "ตัด color shortcut — พิสูจน์ด้วยการวัด: โมเดลเทรนภาพสี วัดบน grayscale แล้ว rust recall ตกเป็น 0"),
    ("ทำไม YOLO11n ไม่ใช่รุ่นใหญ่", "ablation: 11n (0.867) > 11s (0.840) บน split สะอาด — เบากว่า 3 เท่า → model size ไม่ใช่ปัจจัยหลัก"),
    ("Accuracy ที่ได้ดีไหมเทียบคนอื่น", "YOLO มาตรฐานบน NEU-DET ~0.73–0.79, รุ่นปรับปรุง 0.78–0.86 ; เราได้ 0.867 "
     "อยู่ช่วงบน — แต่ย้ำว่าเทียบตรงไม่ได้ (8 คลาส, grayscale, รวม 3 dataset, re-split)"),
    ("จุดขายของโปรเจค", "ไม่ใช่แค่ตัวเลข mAP — แต่เป็น (1) กระบวนการที่ตรวจสอบตัวเองเข้ม (leakage audit, multi-seed, "
     "ablation หลายชั้น) และ (2) ความซื่อสัตย์กับผลลบ (Stage 1 ไม่ช่วย, domain gap) ซึ่งเป็น engineering ที่ดี"),
]


# ============================================================ สร้าง docx

def build():
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    doc.styles["Normal"].font.name = "Tahoma"
    doc.styles["Normal"].font.size = Pt(10.5)

    def h1(t):
        doc.add_page_break()
        doc.add_heading(t, level=1)

    def h2(t):
        doc.add_heading(t, level=2)

    def para(t, bold=False):
        p = doc.add_paragraph()
        r = p.add_run(t)
        r.bold = bold
        return p

    def bullets(seq):
        for s in seq:
            doc.add_paragraph(s, style="List Bullet")

    def kv_bullets(pairs):
        for k, v in pairs:
            p = doc.add_paragraph(style="List Bullet")
            r = p.add_run(k + " — ")
            r.bold = True
            p.add_run(v)

    def table(headers, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(headers))
        try:
            t.style = "Light Grid Accent 1"
        except KeyError:
            t.style = "Table Grid"
        for i, hh in enumerate(headers):
            run = t.rows[0].cells[i].paragraphs[0].add_run(hh)
            run.bold = True
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)
        if widths:
            for r in t.rows:
                for i, w in enumerate(widths):
                    r.cells[i].width = Inches(w)
        return t

    # ---------- ปก ----------
    ti = doc.add_heading("เอกสารเตรียมสอบปกป้องโปรเจค", 0)
    ti.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st = doc.add_paragraph("ระบบตรวจจับตำหนิพื้นผิวเหล็กด้วยการเรียนรู้เชิงลึก (2-Stage: DMS46 → YOLO11)")
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st.runs[0].bold = True
    doc.add_paragraph(
        "สรุปทุกอย่างที่ควรตอบได้ในห้องสอบ — จัดตามกรอบ 8 หัวข้อ (ภาพรวม / architecture / database / "
        "diagram / algorithm-AI / API-backend / security / testing) + เวอร์ชัน AI ที่ใช้ + cheat sheet"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("สร้างอัตโนมัติจาก make_defense_prep_doc.py — อิงโค้ดจริงในโปรเจค"
                      ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------- 0. วิธีใช้ ----------
    h1("0. วิธีใช้เอกสารนี้")
    para("สูตรจำ: ทุก feature ในระบบ ตอบ 7 คำถามนี้ให้ได้", bold=True)
    bullets(SEVEN_Q)
    para("ลำดับความสำคัญที่ควรแม่น: (1) ระบบตัวเอง (2) Architecture (3) Database (4) Diagram "
         "(5) Algorithm/AI — จุดขาย (6) Testing (7) Security", bold=True)
    para("จุดที่กรรมการชอบจับผิดในโปรเจคนี้: ต้อง “เล่าเองได้” ว่า Stage 1 (DMS46) ทดลองแล้ว "
         "ไม่ช่วย (negative ablation) และโมเดลเล่มจบไม่ transfer ไปภาพถ่ายจริง (domain gap) — "
         "สองข้อนี้คือผลลบที่เรารายงานอย่างตั้งใจ ไม่ใช่ความผิดพลาด")

    # ---------- 1. ภาพรวมระบบ ----------
    h1("1. ภาพรวมระบบ")
    h2("1.1 พูด 30–60 วินาที (pitch)")
    para(PITCH)
    h2("1.2 ปัญหา / ผู้ใช้ / ขอบเขต")
    kv_bullets(PROBLEM)
    h2("1.3 ทำงานตั้งแต่ต้นจนจบ (end-to-end)")
    bullets(FLOW_END_TO_END)
    h2("1.4 จุดเด่น")
    bullets([
        "soft-gate + fallback: ทำงานได้แม้ Stage 1 หาโลหะไม่เจอ (ตรวจทั้งภาพแทน)",
        "per-class confidence threshold: แต่ละคลาสตั้งเกณฑ์เองจาก val (macro-F1 0.824 → 0.844)",
        "cross-region NMS: ตัดผลซ้ำข้ามบริเวณที่กรอบทับกัน",
        "แยก “ยืนยัน” กับ “อาจมี” — โหมดความไวปรับได้ แต่ผลที่ผ่านเพราะลดเกณฑ์ไม่ขึ้น “ความเสี่ยงสูง”",
        "กระบวนการตรวจสอบตัวเองเข้ม: leakage audit, multi-seed n=4, ablation หลายชั้น",
    ])

    # ---------- 2. Architecture ----------
    h1("2. Architecture / โครงสร้างระบบ")
    h2("2.1 แผนผัง")
    para(
        "ผู้ใช้ (เบราว์เซอร์)\n"
        "   ↓  อัปโหลด/ถ่ายภาพ  (Gradio event: .click / .change / .stream)\n"
        "app.py  (Gradio server — โปรเซส Python เดียว)\n"
        "   ↓  เรียกฟังก์ชันตรง ๆ (ไม่มี REST)\n"
        "analyze()  →  pipeline.py  (แกน orchestrate)\n"
        "   ├─ Stage 0  steel_gate.py      (YOLO11n-cls — advisory)\n"
        "   ├─ Stage 1  run_stage1()       (DMS46 TorchScript → mask → build_regions + fallback)\n"
        "   └─ Stage 2  run_stage2()×N     (YOLO11 ต่อ crop → per-class threshold) → cross_region_nms()\n"
        "   ↓\n"
        "ภาพผล + verdict + ตาราง  →  กลับหน้าเว็บ  ;  *_result.json / zip  →  ดิสก์"
    )
    h2("2.2 คำถามที่มักโดนถาม (พร้อมคำตอบ)")
    kv_bullets(WHY_ARCH)
    h2("2.3 Deployment")
    para("รันบนโน้ตบุ๊ก Windows 11 เครื่องเดียว: Python 3.11 venv (โค้ด + PyTorch + Ultralytics + DMS46_v1.pt "
         "+ best.pt + thresholds.json), เบราว์เซอร์ต่อ Gradio ที่ 127.0.0.1:7860 (หรือวง LAN), GPU RTX 3050 ผ่าน CUDA 12.4 "
         "ไม่มี server / database / cloud")

    # ---------- 3. Database ----------
    h1("3. Database / การเก็บข้อมูล")
    h2("3.1 ภาพรวม (ต้องตอบให้ได้ว่าทำไมไม่มี DB)")
    para(DB_WHY)
    h2("3.2 เอนทิตีเชิงแนวคิด (ถ้าถามว่ามีตารางอะไรบ้าง)")
    table(["เอนทิตี", "เก็บที่ไหน", "ฟิลด์หลัก"], DB_ENTITIES, widths=[1.4, 1.9, 3.1])
    para("ความสัมพันธ์: " + DB_RELATIONS)
    h2("3.3 ถาม-ตอบ (PK/FK/normalization/ลบข้อมูล)")
    kv_bullets(DB_QA)
    h2("3.4 Relational schema เทียบเท่า (ถ้าย้ายเข้า RDBMS ในอนาคต)")
    table(["ตาราง", "คอลัมน์ (PK/FK ระบุ)"], DB_IF_MIGRATE, widths=[1.9, 4.5])

    # ---------- 4. Diagrams ----------
    h1("4. Diagram ของระบบ")
    para("โปรเจคมีไดอาแกรม 10 แบบ (สร้างด้วย make_uml_doc.py → figures/diagrams/uml_*.png). "
         "ไม่ต้องท่องสัญลักษณ์ — เปิดรูปแล้วชี้อธิบายตามนี้:")
    table(["ไดอาแกรม", "อธิบายอะไร (พูดตามนี้ได้)"], DIAGRAMS, widths=[1.7, 4.7])
    para("หมายเหตุ: Deployment / C4 ทั้งหมดสะท้อนว่า “เครื่องเดียว ไม่มี server/DB” — ตรงกับหัวข้อ 3")

    # ---------- 5. Algorithm / AI ----------
    h1("5. Algorithm / Logic / AI  (จุดขาย — เตรียมแน่นสุด)")
    h2("5.1 Input → Process → Output ต่อขั้นตอน")
    kv_bullets(AI_STAGES)
    h2("5.2 ข้อมูล (Dataset / Split / Leakage)")
    kv_bullets(AI_DATA)
    h2("5.3 เลือกโมเดลอย่างไร + ทำไม")
    kv_bullets(AI_MODEL_CHOICE)
    h2("5.4 นิยาม metric (ต้องอธิบายได้)")
    table(["Metric", "ความหมาย / สูตร"], METRICS_DEF, widths=[1.7, 4.7])
    h2("5.5 ผลต่อคลาส — โมเดลเล่มจบ (train-gray-n2, multi-seed n=4)")
    para("รวมทุกคลาส: mAP50 0.867 ± 0.010 | mAP50-95 0.536 | precision 0.853 | recall 0.809", bold=True)
    table(["คลาส", "mAP50 (mean)", "หมายเหตุ"],
          [(c, m, n) for c, m, n in RESULTS_PERCLASS], widths=[1.6, 1.3, 3.5])
    h2("5.6 Failure analysis — ถ้า AI ทายผิด เกิดจากอะไร")
    kv_bullets(AI_FAILURE)
    h2("5.7 สองรางของโมเดล (ต้องแยกให้ชัดในห้องสอบ)")
    table(["run", "ราง", "เทรนจาก", "ผล"], MODELS_TRACK, widths=[1.3, 1.4, 1.9, 1.8])

    # ---------- 6. API / Backend ----------
    h1("6. API / Backend")
    para("โปรเจคนี้ไม่มี REST API / Flask / Node แยก — ใช้ Gradio ผูก event เข้ากับฟังก์ชัน Python. "
         "ตอบกรรมการตามนี้:")
    kv_bullets(API_POINTS)

    # ---------- 7. Security ----------
    h1("7. Security")
    para("โปรเจคเป็น prototype เครื่องเดียว/วง LAN — ไม่มีชั้น auth โดยตั้งใจ ต้องอธิบายเหตุผลและทางแก้ถ้าจะ deploy จริง:")
    kv_bullets(SEC_POINTS)

    # ---------- 8. Testing ----------
    h1("8. Testing")
    h2("8.1 ระดับการทดสอบในโปรเจค")
    table(["ระดับ", "สคริปต์", "ทดสอบอะไร"], TEST_LEVELS, widths=[1.5, 1.8, 3.1])
    h2("8.2 ตัวอย่าง Test Case (UI)")
    table(["กรณี", "Input", "Expected", "ผล"], TEST_CASES, widths=[1.8, 1.5, 2.1, 1.0])
    h2("8.3 หลักฐานความน่าเชื่อถือของผล")
    bullets([
        "multi-seed n=4 → รายงาน mean ± std ไม่ใช่ตัวเลขเดียว",
        "leakage audit ก่อน/หลัง → ยืนยันว่าตัวเลขไม่ได้มาจากภาพรั่ว",
        "val ≈ test (0.854 ≈ 0.853) → ไม่ overfit",
        "ablation แยกผล: label สะอาด + grayscale (+0.29 mAP50) vs 11n→11s (+0.017)",
    ])

    # ---------- 9. คำถาม "ทำไม" ----------
    h1("9. คำถาม “ทำไม” ที่ต้องซ้อม")
    para("กรรมการถามต่อจาก “ทำอะไร” เป็น “ทำไม” เสมอ — อย่าตอบสั้นแบบ “เพราะง่าย”:")
    kv_bullets(WHY_DRILL)

    # ---------- 10. Key AI versions ----------
    h1("10. เวอร์ชัน / เครื่องมือ AI ที่ใช้")
    table(["ส่วน", "เครื่องมือ / โมเดล", "เวอร์ชัน", "หมายเหตุ"], AI_VERSIONS, widths=[1.5, 1.9, 1.3, 1.7])
    para("ที่มาโมเดล: DMS46 = apple/ml-dms-dataset (Upchurch & Niu, ECCV 2022) ; "
         "YOLO11 = Ultralytics ; น้ำหนักเริ่มต้น yolo11n.pt เทรนบน COCO ; "
         "dataset = NEU-DET + Rust + Crack (Roboflow) รวมเป็น 8 คลาส")

    # ---------- 11. Cheat sheet ----------
    h1("11. ตัวเลข / ข้อเท็จจริงที่ต้องจำเข้าห้องสอบ")
    bullets(CHEAT)

    DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX))
    print("เขียน:", DOCX.relative_to(BASE))


if __name__ == "__main__":
    build()
