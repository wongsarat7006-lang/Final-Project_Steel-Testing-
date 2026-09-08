# -*- coding: utf-8 -*-
"""thesis_content.py — เนื้อหาบทที่ 3 + main สำหรับ make_thesis_doc.build()
แยกไฟล์เพื่อเลี่ยงปัญหา heredoc; เรียกจาก make_thesis_doc.py
เนื้อหาอ้างอิงจากงานร่างเดิมของผู้จัดทำ (docs/งานของผม) + โค้ดจริง + README/thesis_notes
"""
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH


def chapter3(D, figs):
    d = D.d
    BODY_PT = 16
    TH = "TH Sarabun New"

    D.chapter(3, "วิธีการดำเนินงาน")
    D.p("บทนี้นำเสนอขั้นตอนการดำเนินงานวิจัยและการพัฒนาระบบตรวจจับตำหนิผิวเหล็กด้วยปัญญาประดิษฐ์และ "
        "อธิบายได้ ตั้งแต่การกำหนดประชากรและกลุ่มตัวอย่าง การเก็บรวบรวมข้อมูล เครื่องมือที่ใช้ ระยะเวลา "
        "ดำเนินงาน การวิเคราะห์ความต้องการ การออกแบบระบบ การออกแบบฐานข้อมูล ตลอดจนขั้นตอนการพัฒนา "
        "และการทดสอบระบบ")

    # ---------------- 1. ประชากรและกลุ่มตัวอย่าง ----------------
    D.h("1. ประชากรและกลุ่มตัวอย่าง (Population and Sample)", 2)
    D.h("1.1 ชุดข้อมูลภาพสำหรับพัฒนาและทดสอบแบบจำลอง", 3)
    D.bullets([
        "ประชากร: ภาพถ่ายพื้นผิวเหล็กจากชุดข้อมูลมาตรฐาน NEU Surface Defect Database (ตำหนิ 6 ประเภท) "
        "ร่วมกับชุดข้อมูลสนิม (RUST + DANGER-RUST) และชุดข้อมูลรอยแตก ที่จัดการผ่านแพลตฟอร์ม Roboflow",
        "กลุ่มตัวอย่าง: ภาพที่ผ่านการรวม (merge) เป็น 8 คลาส แล้วแบ่งด้วยวิธี group-aware stratified "
        "re-split เป็นชุดฝึก 3,338 ภาพ / ชุดตรวจสอบ 422 ภาพ / ชุดทดสอบ 425 ภาพ (สัดส่วนประมาณ 80/10/10 "
        "และครบทั้ง 8 คลาสในทุกชุด)",
        "ชุดทดสอบภาพถ่ายจริง (real_test): ภาพถ่ายเหล็กระดับสถานที่จำนวน 18 ภาพ ใช้ประเมินช่องว่างเชิง "
        "โดเมนและคุณค่าของขั้นที่ 1 กับภาพที่ไม่ได้ควบคุมสภาพ",
    ])
    D.p("การแบ่งข้อมูลใช้วิธี group-aware เพื่อป้องกันการรั่วไหลของข้อมูล (Data Leakage) เนื่องจากชุด "
        "ข้อมูลสนิมบางส่วนเป็นภาพถ่ายต่อเนื่อง (burst) ที่เกือบเหมือนกัน หากสุ่มแยกเฟรมติดกันไปคนละชุดจะ "
        "ทำให้ผลการวัดสูงเกินจริง")

    D.h("1.2 ผู้ประเมินและทดสอบระบบ (System Evaluators)", 3)
    D.bullets([
        "ผู้เชี่ยวชาญ/ผู้มีประสบการณ์ด้านการตรวจสอบคุณภาพหรือการพัฒนาระบบ จำนวน 3–5 คน สำหรับประเมิน "
        "คุณภาพระบบ (Functional Correctness, Performance, Explainability, Data Security)",
        "ผู้ใช้งานทั่วไป จำนวน 15–30 คน สำหรับประเมินความง่ายในการใช้งาน (Usability) และความพึงพอใจต่อ "
        "หน้าเว็บและการแสดงผล",
    ])

    # ---------------- 2. การเก็บรวบรวมข้อมูล ----------------
    D.h("2. การเก็บรวบรวมข้อมูล (Data Collection)", 2)
    D.h("2.1 การเก็บและเตรียมชุดข้อมูลภาพ", 3)
    D.p("ขั้นตอนการเตรียมข้อมูลตั้งแต่ภาพดิบจนถึงพร้อมฝึก แสดงในภาพต่อไปนี้")
    D.figure(figs["dataprep"], "ขั้นตอนการเตรียมชุดข้อมูล (raw → merge → resplit → fix labels → grayscale → oversample → train)", width=6.3)
    D.numbered([
        "รวมชุดข้อมูล 3 แหล่งเป็น 8 คลาส (merge_datasets.py) — กรองไฟล์ปนเปื้อนด้วย prefix ของชื่อไฟล์ "
        "แปลง class id ให้ตรงกัน และแปลง label แบบ polygon ของรอยแตกให้เป็น bounding box",
        "แบ่ง train/valid/test ใหม่แบบ stratified และ group-aware (resplit_grouped.py)",
        "แก้ป้ายกำกับ (fix_labels.py) — รวมกล่องย่อยของ crazing และ rolled-in_scale เป็นกล่องเดียวต่อภาพ "
        "และตัดกล่องที่ผิดรูป โดยสำรองไฟล์เดิมไว้",
        "สร้างชุดข้อมูลระดับเทา (make_grayscale_dataset.py) เพื่อลด shortcut ที่แบบจำลองใช้โทนสีแยกโดเมน",
        "ทำ class-balanced oversampling (make_oversampled_list.py) สำหรับคลาสที่มีตัวอย่างน้อย",
        "ฝึกแบบจำลองขั้นที่ 2 ด้วย train.py (YOLO11n, recipe texture) บนชุดข้อมูลระดับเทา",
    ])
    D.p("การกระจายของจำนวนตัวอย่างในแต่ละคลาสแสดงในภาพต่อไปนี้ ซึ่งชี้ให้เห็นความไม่สมดุลของคลาส (Class "
        "Imbalance) ที่ต้องแก้ด้วยการ oversampling")
    D.figure(figs["classdist"], "การกระจายของคลาสตำหนิในชุดข้อมูลรวม", width=6.0)

    D.h("2.2 การเก็บข้อมูลแบบสอบถามความพึงพอใจ", 3)
    D.p("เก็บข้อมูลจากกลุ่มตัวอย่างผู้ใช้งานหลังทดลองใช้ระบบต้นแบบ ด้วยแบบสอบถามมาตราส่วนประมาณค่า 5 "
        "ระดับ (Likert Scale) ครอบคลุมด้านความถูกต้องของฟังก์ชัน ความเร็ว ความเข้าใจง่ายของผลลัพธ์ "
        "(Explainability) และความง่ายในการใช้งานหน้าเว็บ")

    # ---------------- 3. เครื่องมือในการพัฒนาระบบ ----------------
    D.h("3. เครื่องมือในการพัฒนาระบบ (Development Tools)", 2)
    D.table(["ประเภท", "รายการ"], [
        ["ฮาร์ดแวร์", "โน้ตบุ๊ก Windows 11, หน่วยประมวลผลกราฟิก NVIDIA RTX 3050 6GB (CUDA 12.4), "
                     "กล้องดิจิทัล/เว็บแคมสำหรับถ่ายภาพชิ้นงาน"],
        ["ภาษาและไลบรารี", "Python 3.11, PyTorch 2.6, Ultralytics YOLO 8.4, OpenCV, NumPy, Pillow"],
        ["แบบจำลอง", "DMS46 (Apple Dense Material Segmentation, TorchScript) — Stage 1 ; "
                    "YOLO11n ฝึกเอง (run: train-gray-n2) — Stage 2"],
        ["หน้าเว็บ/แสดงผล", "Gradio (หน้าเว็บต้นแบบ), matplotlib (รูปประกอบ), python-docx (เอกสาร)"],
        ["จัดการข้อมูล", "Roboflow (จัดการชุดข้อมูล/ติดป้ายกำกับ/Augmentation)"],
        ["ควบคุมเวอร์ชัน", "Git, GitHub"],
        ["Future Work", "SQLite/PostgreSQL (ฐานข้อมูลบันทึกผล), Streamlit/React (แดชบอร์ด), "
                        "LINE Messaging API (การแจ้งเตือน)"],
    ], "เครื่องมือและเทคโนโลยีที่ใช้ในการพัฒนาระบบ", col_w=[3.2, 11.5])

    # ---------------- 4. ระยะเวลาในการศึกษา ----------------
    D.h("4. ระยะเวลาในการศึกษา", 2)
    D.table(["สัปดาห์", "ช่วงเวลา", "กิจกรรมหลัก"], [
        ["1", "15 – 21 ก.ค. 2569", "ศึกษาทฤษฎี XAI / Computer Vision, เลือกแบบจำลอง, กำหนดขอบเขต"],
        ["2", "22 – 28 ก.ค. 2569", "ศึกษาการนำไปใช้เชิงอุตสาหกรรม, หาโครงระบบต้นแบบอ้างอิง"],
        ["3", "29 ก.ค. – 4 ส.ค. 2569", "วิเคราะห์ข้อดี-ข้อเสียของระบบเดิม, กำหนดจุดเด่นและกลุ่มผู้ใช้"],
        ["4", "5 – 11 ส.ค. 2569", "ออกแบบสถาปัตยกรรม, เตรียมและติดป้ายกำกับชุดข้อมูล"],
        ["5", "12 – 18 ส.ค. 2569", "ฝึกแบบจำลองขั้นที่ 2, วัดผล mAP/F1, ปรับ per-class threshold"],
        ["6", "19 – 25 ส.ค. 2569", "ตรวจสอบ Data Leakage, แบ่งข้อมูลใหม่, ฝึกซ้ำหลายรอบ (multi-seed)"],
        ["7", "26 – 31 ส.ค. 2569", "พัฒนาหน้าเว็บ, ทดสอบ End-to-End, สรุปผลและจัดทำรายงาน"],
    ], "แผนการดำเนินงานของโครงงาน", col_w=[1.6, 4.0, 9.0])

    # ---------------- 5. การวิเคราะห์ความต้องการ ----------------
    D.h("5. การวิเคราะห์ความต้องการ (Requirements Analysis)", 2)
    D.h("5.1 ข้อกำหนดฟังก์ชันการทำงาน (Functional Requirements: FR)", 3)
    D.table(["รหัส", "ความต้องการ", "รายละเอียด", "ความสำคัญ"], [
        ["FR-01", "รับภาพนำเข้า", "อัปโหลดภาพเหล็ก (jpg/jpeg/png/bmp/webp/tif) ผ่านหน้าเว็บ หรือระบุ "
                  "--image / --folder ผ่าน CLI", "สูง"],
        ["FR-02", "Stage 1 — ระบุบริเวณโลหะ", "ตรวจหาพื้นที่วัสดุโลหะด้วย DMS46 แล้วแปลงเป็นกรอบบริเวณ "
                  "(mask → contour → merge)", "สูง"],
        ["FR-03", "Fallback ตรวจทั้งภาพ", "ถ้าพื้นที่โลหะ < 5% หรือไม่พบบริเวณ ให้เพิ่มกรอบ = ทั้งภาพ", "สูง"],
        ["FR-04", "Stage 2 — ตรวจจับตำหนิ", "ตรวจจับและจำแนกตำหนิ 8 ประเภทในแต่ละบริเวณด้วย YOLO11", "สูง"],
        ["FR-05", "ปรับ Confidence threshold", "ผู้ใช้ปรับ threshold รวม (0.1–0.9) และระบบใช้ per-class "
                  "threshold จาก thresholds.json อัตโนมัติ", "กลาง"],
        ["FR-06", "โหมดตรวจละเอียด (TTA)", "เปิด test-time augmentation เพื่อเพิ่ม recall (ช้าลง 2–3 เท่า)", "ต่ำ"],
        ["FR-07", "Cross-region NMS", "ตัด detection ซ้ำข้ามบริเวณ (คลาสเดียวกัน, IoU > 0.5 ในพิกัดภาพเต็ม)", "กลาง"],
        ["FR-08", "แสดงผลบนหน้าเว็บ", "กรอบบริเวณ + กรอบตำหนิ + ป้ายไทย + ระดับความเสี่ยง + ตารางรายการ "
                  "+ ภาพผล Stage 1 + สรุปผล", "สูง"],
        ["FR-09", "บันทึกผลลัพธ์ (CLI)", "เขียน <name>_result.jpg และ <name>_result.json ต่อภาพ; "
                  "โหมด --folder สร้าง _index.json", "สูง"],
        ["FR-10", "รองรับภาพหลายรูปแบบ", "แปลง grayscale/RGBA เป็น BGR; แปลง crop เป็น grayscale "
                  "อัตโนมัติเมื่อแบบจำลองฝึกบน grayscale", "กลาง"],
        ["FR-11", "เตรียมข้อมูล (ผู้พัฒนา)", "merge → resplit → fix_labels → grayscale → oversample", "สูง"],
        ["FR-12", "ฝึกและวัดผล (ผู้พัฒนา)", "train.py, evaluate.py, evaluate_stage1.py, evaluate_real.py, "
                  "tune_thresholds.py", "สูง"],
        ["FR-13", "อธิบายผล/ประเมินความรุนแรง (Future Work)", "คำนวณ Defect Area Ratio → Severity "
                  "(Low/Med/High) → Pass/Fail → บันทึก Log → แดชบอร์ด/แจ้งเตือน", "กลาง"],
    ], "ข้อกำหนดฟังก์ชันการทำงานของระบบ", col_w=[1.5, 3.4, 8.3, 1.5])

    D.h("5.2 ข้อกำหนดคุณสมบัติทั่วไป (Non-Functional Requirements: NFR)", 3)
    D.table(["รหัส", "ด้าน", "เกณฑ์ / รายละเอียด"], [
        ["NFR-01", "Performance", "โหลดแบบจำลองครั้งเดียวตอนเริ่ม; ต่อภาพบน GPU: Stage 1 ~208 มิลลิวินาที, "
                   "Stage 2 ~10 มิลลิวินาที/บริเวณ"],
        ["NFR-02", "Hardware / Platform", "Python 3.11; NVIDIA GPU ≥ 6 GB (RTX 3050) แนะนำ; รันบน CPU ได้แต่ช้า"],
        ["NFR-03", "Accuracy", "Stage 2 (train-gray-n2) ชุดทดสอบ mAP@0.5 0.867 ± 0.010 (n=4), "
                   "mAP@0.5–0.95 0.536, recall 0.809; เน้น recall; crack/crazing อ่อนสุด"],
        ["NFR-04", "Usability", "หน้าเว็บภาษาไทย ขั้นตอนเดียว: อัปโหลด → ตรวจ → เห็นผล; มีตัวบ่งชี้ความคืบหน้า "
                   "และภาพตัวอย่างให้ทดลอง"],
        ["NFR-05", "Portability", "รันบนเครื่องเดียว ไม่มี server / ฐานข้อมูล / cloud; ต่อเน็ตเฉพาะครั้งแรก "
                   "(ดาวน์โหลด weights)"],
        ["NFR-06", "Maintainability", "โค้ดแยกโมดูล (pipeline เป็นแกน), ตรรกะ fallback รวมที่เดียว, "
                   "มี test_smoke.py, สคริปต์เตรียมข้อมูลทำซ้ำได้ (seed คงที่)"],
        ["NFR-07", "Reliability", "fallback เมื่อ Stage 1 หาโลหะไม่พบ; ป้องกันภาพเปิดไม่ได้/ crop เล็กเกินไป"],
        ["NFR-08", "Compatibility (train/serve)", "ตรวจอัตโนมัติว่าแบบจำลองฝึกบน grayscale แล้วปรับ input "
                   "ให้ตรงกัน กัน train/serve skew"],
        ["NFR-09", "Security / Privacy", "ประมวลผลบนเครื่อง ไม่ส่งภาพออกนอก; ไม่มีบัญชีผู้ใช้ (ต้นแบบ)"],
        ["NFR-10", "Scope / Limitation", "เป็นต้นแบบผู้ช่วยคัดกรอง ไม่รองรับเรียลไทม์บนสายพาน ไม่ต่อ PLC"],
    ], "ข้อกำหนดคุณสมบัติทั่วไปของระบบ", col_w=[1.6, 3.0, 10.0])

    # ---------------- 6. การออกแบบระบบ ----------------
    D.h("6. การออกแบบระบบ (System Design)", 2)

    D.h("6.1 สถาปัตยกรรมระบบ (System Architecture)", 3)
    D.p("ระบบมีอินเทอร์เฟซ 2 ทาง คือ หน้าเว็บ (Gradio, app.py) และคำสั่งบรรทัด (CLI, pipeline.py) "
        "ซึ่งทั้งคู่เรียกใช้แกนประมวลผลเดียวกัน โครงสร้างโดยรวมและรายละเอียดภายในแต่ละขั้นแสดงในภาพต่อไปนี้")
    D.figure(figs["arch"], "สถาปัตยกรรมระบบสองขั้น (ภาพรวม)", width=5.6)
    D.bullets([
        "SA-01 อินเทอร์เฟซ 2 ทาง: Web UI (Gradio) และ CLI — เรียกแกนประมวลผลตัวเดียวกัน",
        "SA-02 Pipeline Core (pipeline.py): โหลดแบบจำลอง, เรียก Stage 1/2, ทำ NMS, กรอง threshold, วาด/บันทึกผล",
        "SA-03 Stage 1 — DMS46: resize ด้านยาว 512 + ImageNet normalize → label map 46 วัสดุ → "
        "เลือก index 22 (\"Metal\") → binary mask",
        "SA-04 Region extraction: morphology + findContours + กรองด้วยพื้นที่/สัดส่วน + รวมกรอบที่อยู่ติดกัน",
        "SA-05 Fallback: ถ้า metal_ratio < 0.05 หรือไม่พบกรอบ → เพิ่มกรอบ = ทั้งภาพ",
        "SA-06 Stage 2 — YOLO11n ฝึกเองบนชุดข้อมูลรวม 8 คลาส (grayscale) รันตรวจตำหนิบนแต่ละบริเวณ",
        "SA-07 Grayscale guard: ตรวจจาก checkpoint ว่าฝึกบน grayscale หรือไม่ ถ้าใช่แปลง crop เป็น "
        "grayscale ก่อน predict",
        "SA-08 Cross-region NMS: แปลง bbox เป็นพิกัดภาพเต็ม แล้วตัด detection ซ้ำข้ามบริเวณ (IoU > 0.5)",
        "SA-09 Per-class threshold: กรอง detection ที่ค่าความมั่นใจต่ำกว่าเกณฑ์รายคลาสจาก thresholds.json",
        "SA-10 Output: วาดกรอบ + ป้ายไทย + ระดับความเสี่ยง → หน้าเว็บ หรือไฟล์ *_result.jpg / *_result.json",
        "SA-11 External: DMS46_v1.pt (Apple), Ultralytics YOLO runtime, ระบบไฟล์ท้องถิ่น — ไม่มี API เครือข่าย",
        "SA-12 Deployment: เครื่องเดียว (Windows 11 + RTX 3050 6GB, Python 3.11) — ไม่มี server/ฐานข้อมูล/cloud",
    ])
    D.figure(figs["stage1"], "ขั้นที่ 1 — DMS46 Metal Localization (ภายใน)", width=6.3)
    D.figure(figs["stage2"], "ขั้นที่ 2 — YOLO11 Defect Detection (ภายใน)", width=6.3)

    D.h("6.2 มุมมองสถาปัตยกรรม C4 และการติดตั้ง (Deployment)", 3)
    D.figure(figs["c4l1"], "C4 Model — ระดับที่ 1: System Context", width=6.2)
    D.figure(figs["c4l2"], "C4 Model — ระดับที่ 2: Container", width=6.4)
    D.figure(figs["deploy"], "แผนภาพการติดตั้ง (Deployment Diagram) — รันบนเครื่องเดียว", width=6.2)

    D.h("6.3 แผนผังกระบวนการทำงาน (Flowchart)", 3)
    D.p("แผนผังลำดับขั้นตอนการทำงานของระบบหลักและขั้นตอนการปรับแต่งภาพแสดงในภาพต่อไปนี้")
    D.figure(figs["flowmain"], "แผนผังลำดับขั้นตอนการทำงานของระบบหลัก", width=5.6)
    D.figure(figs["flowpre"], "แผนผังลำดับขั้นตอนการปรับแต่งภาพ (Pre-processing)", width=5.8)

    D.h("6.4 แผนภาพกรณีการใช้งาน (Use Case Diagram)", 3)
    D.figure(figs["usecase"], "แผนภาพกรณีการใช้งานของระบบตรวจจับตำหนิผิวเหล็ก", width=6.2)
    D.bullets([
        "UC-01 ผู้ใช้งาน — อัปโหลดภาพเหล็กเข้าระบบ",
        "UC-02 ผู้ใช้งาน — ปรับ confidence threshold / เปิดโหมดตรวจละเอียด",
        "UC-03 ผู้ใช้งาน — สั่งตรวจภาพ (ระบบรัน Stage 1 + Stage 2)  «include» UC-01",
        "UC-04 ผู้ใช้งาน — ดูผล: กรอบ + ชนิดตำหนิ + ระดับความเสี่ยง + ภาพผล Stage 1  «include» UC-03",
        "UC-05 ผู้ใช้งาน — บันทึก/ส่งออกผล (ภาพ + JSON)",
        "UC-06 ผู้พัฒนา — เตรียมชุดข้อมูล (merge → resplit → fix_labels → grayscale → oversample)",
        "UC-07 ผู้พัฒนา — ฝึกแบบจำลอง Stage 2 (train.py)",
        "UC-08 ผู้พัฒนา — วัดผล mAP / F1 (evaluate.py)",
        "UC-09 ผู้พัฒนา — หา per-class confidence threshold (tune_thresholds.py)",
        "UC-10 ผู้พัฒนา — วัดผล Stage 1 เชิงตัวเลข / วัดกับภาพถ่ายจริง (evaluate_stage1.py, evaluate_real.py)",
        "UC-11 ผู้พัฒนา — สร้างรูป/เอกสารประกอบ (make_figures / make_*_doc)",
    ])

    D.h("6.5 คำอธิบายกรณีการใช้งาน (Use Case Description)", 3)
    ucs = [
        ("UC-03 สั่งตรวจภาพ",
         "ผู้ใช้งาน", "มีภาพนำเข้าในระบบแล้ว (UC-01)",
         ["ผู้ใช้กดปุ่ม \"ตรวจสอบ\"",
          "ระบบเตรียมภาพ (แปลงช่องสี/ระดับเทา ตามแบบจำลอง)",
          "ระบบเรียก Stage 1 หาบริเวณโลหะ และใช้ fallback ถ้าจำเป็น",
          "ระบบเรียก Stage 2 ตรวจจับตำหนิในแต่ละบริเวณ",
          "ระบบรวมผลข้ามบริเวณด้วย Cross-region NMS และกรองด้วย per-class threshold"],
         "ระบบแสดงกรอบตำหนิ ป้ายชนิด ค่าความมั่นใจ ระดับความเสี่ยง และภาพผล Stage 1"),
        ("UC-04 ดูผลการตรวจ",
         "ผู้ใช้งาน", "ระบบประมวลผลภาพเสร็จแล้ว (UC-03)",
         ["ระบบแสดงภาพผลพร้อมกรอบและป้ายกำกับภาษาไทย",
          "ระบบแสดงตารางรายการตำหนิเรียงตามระดับความเสี่ยง",
          "ผู้ใช้ตรวจสอบภาพบริเวณโลหะ (Stage 1) และสรุปผลประกอบ"],
         "ผู้ใช้เข้าใจว่าพบตำหนิชนิดใด ที่ตำแหน่งใด และมีความเสี่ยงระดับใด"),
        ("UC-07 ฝึกแบบจำลอง Stage 2",
         "ผู้พัฒนา", "เตรียมชุดข้อมูลระดับเทาและไฟล์ oversample แล้ว (UC-06)",
         ["ผู้พัฒนาเรียก train.py พร้อมพารามิเตอร์ (recipe, model, epochs)",
          "ระบบฝึกแบบจำลองและตรวจสอบบนชุด validation อัตโนมัติ",
          "ระบบบันทึก weights (best.pt) และกราฟการฝึก"],
         "ได้แบบจำลอง Stage 2 ที่พร้อมนำไปวัดผลและใช้งานใน pipeline"),
        ("UC-10 วัดผลกับภาพถ่ายจริง",
         "ผู้พัฒนา", "มีโฟลเดอร์ real_test/ พร้อม labels.csv",
         ["ผู้พัฒนาเรียก evaluate_real.py",
          "ระบบรัน pipeline (มี Stage 1) และ baseline (ไม่มี Stage 1) บนภาพชุดเดียวกัน",
          "ระบบคำนวณ micro/macro Precision, Recall, F1 และเวลาต่อภาพ"],
         "ได้ตารางเปรียบเทียบ pipeline กับ baseline เพื่อประเมินคุณค่าของ Stage 1"),
    ]
    for name, actor, pre, steps, post in ucs:
        D.table(["หัวข้อ", "รายละเอียด"], [
            ["ชื่อกรณีการใช้งาน", name],
            ["ผู้กระทำ (Actor)", actor],
            ["เงื่อนไขก่อนเริ่ม", pre],
            ["ลำดับเหตุการณ์หลัก", "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))],
            ["ผลลัพธ์ที่ได้", post],
        ], f"Use Case Description : {name}", col_w=[3.4, 11.3])

    D.h("6.6 แผนภาพลำดับ (Sequence Diagram)", 3)
    D.p("แผนภาพลำดับแสดงการโต้ตอบระหว่างวัตถุ/โมดูลเมื่อผู้ใช้กดปุ่ม \"ตรวจสอบ\" บนหน้าเว็บ ตั้งแต่การ "
        "โหลดแบบจำลองครั้งแรก การเรียก Stage 1 (DMS46) การสร้างบริเวณและ fallback การวนเรียก Stage 2 "
        "(YOLO11) ต่อบริเวณ การรวมผลด้วย Cross-region NMS จนถึงการวาดผลและส่งค่ากลับหน้าเว็บ")
    D.figure(figs["sequence"], "แผนภาพลำดับของฟังก์ชัน analyze() เมื่อผู้ใช้สั่งตรวจภาพ", width=6.4)

    D.h("6.7 แผนภาพกิจกรรมและสถานะ (Activity & State Diagram)", 3)
    D.figure(figs["activity"], "แผนภาพกิจกรรมของการประมวลผล 1 ภาพ (process_image)", width=4.0)
    D.figure(figs["state"], "แผนภาพสถานะ — วงจรชีวิตของภาพระหว่างการประมวลผล", width=6.3)

    D.h("6.8 แผนภาพคลาส (Class Diagram)", 3)
    D.p("ระบบพัฒนาแบบ functional โดยแต่ละโมดูล (pipeline, app, evaluate*) มีลักษณะเป็นชุดฟังก์ชันสาธารณะ "
        "(stereotype «utility») และ dict payload (Detection, RegionMeta, Summary) ที่ส่งต่อระหว่างฟังก์ชัน "
        "ดังแผนภาพคลาสต่อไปนี้")
    D.figure(figs["class"], "แผนภาพคลาสแสดงความสัมพันธ์ของโมดูลและโครงสร้างข้อมูล", width=6.0)

    D.h("6.9 การออกแบบส่วนติดต่อผู้ใช้ (User Interface Design)", 3)
    D.p("หน้าเว็บ (Gradio) ออกแบบให้มีขั้นตอนเดียว เน้นใช้งานง่าย: อัปโหลดหรือวางภาพ ระบบตรวจให้อัตโนมัติ "
        "แล้วแสดงการ์ดสรุปผลและตารางรายการตำหนิที่คอลัมน์ขวา ส่วนภาพผลลัพธ์และภาพผลของ Stage 1 แสดงเต็ม "
        "ความกว้างด้านล่าง ตัวปรับตั้งค่าทั้งหมด (โมเดล โหมดความไว ค่า confidence โหมดตรวจละเอียด และ "
        "Stage 0) ถูกยุบไว้ในส่วน “ตัวเลือกขั้นสูง” โครงหน้าจอโดยรวมแสดงในภาพต่อไปนี้")
    D.figure(figs["wireframe2"], "โครงหน้าจอ (Wireframe) ของหน้าเว็บระบบ", width=6.6)
    D.p("ตัวอย่างผลการตรวจจริงของระบบต้นแบบกับตำหนิแต่ละชนิด โดยระบบวาดกรอบตำหนิสีแดง ป้ายชนิดภาษาไทย "
        "และค่าความมั่นใจกำกับไว้บนภาพ")
    D.figure("res_crazing.jpg", "ตัวอย่างผลการตรวจ — รอยแตกลายงา (crazing) บนภาพจากชุด benchmark", width=5.4)
    D.figure("res_scratches.jpg", "ตัวอย่างผลการตรวจ — รอยขีดข่วน (scratches)", width=5.4)
    D.figure("res_patches.jpg", "ตัวอย่างผลการตรวจ — รอยแผ่น/ผิวลอก (patches)", width=5.4)
    D.figure("res_rust.jpg", "ตัวอย่างผลการตรวจ — สนิม (rust) บนภาพถ่ายเหล็กเส้นระดับสถานที่", width=5.0)
    D.figure("res_rust2.jpg", "ตัวอย่างผลการตรวจ — สนิม (rust) บนท่อเหล็กระดับสถานที่", width=6.0)
    D.p("ภาพต่อไปนี้เปรียบเทียบผลของแบบจำลองสองแบบบนภาพถ่ายจริงภาพเดียวกัน — แบบจำลองที่รายงานผลในเล่ม "
        "(ฝึกบน grayscale) กับแบบจำลองที่เพิ่มภาพถ่ายจริงเข้าไปในการฝึก (งานพัฒนาต่อ) เพื่อแสดงช่องว่าง "
        "เชิงโดเมน")
    D.figure("res_thesis_model.jpg", "ผลบนภาพถ่ายจริง — แบบจำลองที่ฝึกบน grayscale (ไม่พบตำหนิ)", width=5.0)
    D.figure("res_adapted_model.jpg", "ผลบนภาพถ่ายจริง — แบบจำลองที่เพิ่มภาพถ่ายจริงในการฝึก (ตรวจพบสนิม)", width=5.0)

    # ---------------- 7. การออกแบบฐานข้อมูล ----------------
    D.h("7. การออกแบบฐานข้อมูล (Database Design)", 2)
    D.p("ระบบต้นแบบปัจจุบันไม่ใช้ระบบจัดการฐานข้อมูล (DBMS) แต่จัดเก็บข้อมูลเป็นไฟล์ JSON / ไฟล์ label "
        "รูปแบบ YOLO / ไฟล์ CSV เนื่องจากเป็นระบบเครื่องเดียวและประมวลผลทีละคำขอ อย่างไรก็ตาม เอกสารนี้ "
        "จัดโครงสร้างเชิงฐานข้อมูลไว้เพื่อรองรับการพัฒนาเป็น RDBMS ในอนาคต")

    D.h("7.1 แบบจำลองข้อมูลเชิงแนวคิด (Conceptual Data Model / ERD)", 3)
    D.figure(figs["erd"], "แบบจำลองข้อมูลเชิงแนวคิด (Entity-Relationship Diagram)", width=6.4)
    D.bullets([
        "RESULT — ไฟล์ *_result.json ต่อภาพ: image, metal_regions, metal_area_ratio, "
        "fallback_full_image, regions[]",
        "REGION — ฝังใน RESULT.regions[]: region_id, box_xywh[4], detections[]",
        "DETECTION — ฝังใน REGION.detections[]: class, confidence, bbox_xywh, bbox_xyxy_crop, "
        "bbox_xyxy_global, region_id",
        "DEFECT_CLASS — คงที่ในโค้ด: id 0–7, name, name_th, risk",
        "DATASET_IMAGE — ไฟล์ภาพฝึก: filename, split {train/valid/test}",
        "LABEL_BOX — ไฟล์ label .txt (YOLO): class_id, xc, yc, w, h (normalized)",
        "REAL_TEST_LABEL — real_test/labels.csv: filename, classes (ระดับภาพ คั่นด้วย ;)",
        "ความสัมพันธ์: RESULT 1–N REGION ; REGION 1–N DETECTION ; DETECTION N–1 DEFECT_CLASS ; "
        "DATASET_IMAGE 1–N LABEL_BOX ; LABEL_BOX N–1 DEFECT_CLASS",
    ])

    D.h("7.2 พจนานุกรมข้อมูล (Data Dictionary)", 3)
    D.table(["เอนทิตี", "ฟิลด์", "ชนิดข้อมูล", "คำอธิบาย"], [
        ["RESULT", "image", "string (path)", "path ของภาพต้นฉบับที่ตรวจ"],
        ["RESULT", "metal_regions", "int", "จำนวนบริเวณที่ Stage 1 พบ (รวมกรอบ fallback ถ้ามี)"],
        ["RESULT", "metal_area_ratio", "float 0–1", "สัดส่วนพื้นที่ที่ระบุว่าเป็นโลหะต่อพื้นที่ภาพ"],
        ["RESULT", "fallback_full_image", "bool", "true ถ้า metal_area_ratio < 0.05 หรือไม่พบบริเวณ"],
        ["REGION", "region_id", "int (PK)", "ลำดับบริเวณในภาพนี้ (เริ่มที่ 0)"],
        ["REGION", "box_xywh[4]", "int×4", "กรอบบริเวณในพิกัดภาพเต็ม (x, y, w, h)"],
        ["DETECTION", "class", "string (FK→DEFECT_CLASS.name)", "ชื่อคลาสภาษาอังกฤษ 1 ใน 8 ประเภท"],
        ["DETECTION", "confidence", "float 0–1", "ค่าความมั่นใจหลังกรอง per-class threshold"],
        ["DETECTION", "bbox_xyxy_global[4]", "float×4", "กรอบตำหนิในพิกัดภาพเต็ม"],
        ["DETECTION", "region_id", "int (FK→REGION.region_id)", "อ้างอิงกลับไปยังบริเวณต้นทาง"],
        ["DEFECT_CLASS", "id / name / name_th / risk", "int / str / str / enum",
         "ลำดับ, ชื่ออังกฤษ, ชื่อไทย, ระดับความเสี่ยง"],
        ["DATASET_IMAGE", "filename / split", "string (PK) / enum", "ชื่อไฟล์, ชุด train/valid/test"],
        ["LABEL_BOX", "class_id / xc,yc,w,h", "int / float×4 (0–1)", "คลาสและพิกัด normalized ตามรูปแบบ YOLO"],
        ["REAL_TEST_LABEL", "filename / classes", "string / multi-value (คั่น ;)",
         "ชื่อไฟล์, ชนิดตำหนิระดับภาพ"],
    ], "พจนานุกรมข้อมูลของเอนทิตีหลักในระบบ", col_w=[2.6, 3.4, 3.6, 5.1])

    D.h("7.3 ตัวอย่างข้อมูลจริง (Sample Data)", 3)
    D.p("ผลลัพธ์จริงจากการรัน pipeline.py บนภาพ rust_example.jpg (พบ 1 บริเวณโลหะ และ 1 ตำหนิ — rust "
        "ความมั่นใจประมาณ 92.8%):")
    code = d.add_paragraph(
        '{ "image": "test_images/rust_example.jpg", "metal_regions": 1, '
        '"metal_area_ratio": 0.4466, "fallback_full_image": false,\n'
        '  "regions": [ { "region_id": 0, "box_xywh": [70,12,324,404],\n'
        '    "detections": [ { "class": "rust", "confidence": 0.9278,\n'
        '      "bbox_xyxy_global": [78.0,12.4,349.4,363.6], "region_id": 0 } ] } ] }')
    code.paragraph_format.left_indent = Cm(0.6)
    for r in code.runs:
        r.font.name = "Consolas"; r.font.size = Pt(12)

    D.h("7.4 Relational Schema เทียบเท่า (แนวทางเมื่อย้ายไปใช้ RDBMS)", 3)
    D.table(["ตาราง", "คอลัมน์ (PK/FK)"], [
        ["RESULT", "result_id (PK), image_path, metal_regions, metal_area_ratio, "
                   "fallback_full_image, created_at"],
        ["REGION", "region_id (PK), result_id (FK→RESULT), box_x, box_y, box_w, box_h"],
        ["DETECTION", "detection_id (PK), region_id (FK→REGION), class_id (FK→DEFECT_CLASS), "
                      "confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2"],
        ["DEFECT_CLASS", "class_id (PK), name, name_th, risk"],
        ["DATASET_IMAGE", "image_id (PK), filename (UNIQUE), split"],
        ["LABEL_BOX", "label_id (PK), image_id (FK→DATASET_IMAGE), class_id (FK→DEFECT_CLASS), "
                      "xc, yc, w, h"],
        ["REAL_TEST_IMAGE", "image_id (PK), filename (UNIQUE)"],
        ["REAL_TEST_LABEL_CLASS", "image_id (FK), class_id (FK)  — PK คู่ (junction table)"],
    ], "Relational Schema (3NF) สำหรับการพัฒนาต่อในอนาคต", col_w=[3.4, 11.3])
    D.p("หมายเหตุการทำ Normalization: ฟิลด์ classes ใน REAL_TEST_LABEL ปัจจุบันเก็บหลายค่าคั่นด้วย “;” "
        "ในไฟล์เดียว (ผิดหลัก 1NF) เพราะออกแบบให้ผู้ใช้แก้ด้วย Excel/Notepad ได้ง่าย เมื่อย้ายเข้า RDBMS "
        "ควรแยกเป็น REAL_TEST_IMAGE และ REAL_TEST_LABEL_CLASS (junction table) เพื่อสอบถามระดับคลาสได้ตรง")

    # ---------------- 8. ขั้นตอนการพัฒนาและการทดสอบ ----------------
    D.h("8. ขั้นตอนการพัฒนาและการทดสอบ (Development & Testing)", 2)
    D.h("8.1 ลำดับขั้นตอนการพัฒนา (Development Pipeline)", 3)
    D.numbered([
        "เตรียมชุดข้อมูล: รวม 3 แหล่ง → แบ่งใหม่ → แก้ป้ายกำกับ → ทำระดับเทา → oversampling",
        "ฝึกแบบจำลอง Stage 2: YOLO11n บนชุดข้อมูลระดับเทา (recipe texture) และฝึกซ้ำหลาย seed เพื่อรายงาน "
        "ค่าเฉลี่ย ± ส่วนเบี่ยงเบนมาตรฐาน",
        "หา per-class confidence threshold จากชุด validation (tune_thresholds.py → thresholds.json)",
        "ประกอบ pipeline สองขั้นและทดสอบ smoke test (test_smoke.py)",
        "วัดผล: mAP รายคลาสบนชุดทดสอบ, วัด Stage 1 เชิงตัวเลข, และวัดกับภาพถ่ายจริง (pipeline vs baseline)",
        "พัฒนาหน้าเว็บ (Gradio) สำหรับอัปโหลดภาพและแสดงผลอธิบายได้",
        "Future Work: เพิ่มการคำนวณ Severity, การตัดสิน Pass/Fail, ฐานข้อมูลบันทึกผล, แดชบอร์ด และการแจ้งเตือน",
    ])

    D.h("8.2 แผนการทดสอบระบบ (Test Plan & Test Cases)", 3)
    D.table(["รหัส", "รายการทดสอบ", "ข้อมูลนำเข้า", "ผลลัพธ์ที่คาดหวัง"], [
        ["TC-01", "ตรวจผิวเหล็กปกติ", "ภาพผิวเหล็กสมบูรณ์ ไม่มีตำหนิ",
         "ไม่พบกรอบตำหนิ → สรุปผล \"ไม่พบตำหนิ\""],
        ["TC-02", "ตรวจตำหนิขนาดเล็ก", "ภาพมีรอยขีดข่วนขนาดเล็ก",
         "พบกรอบตำหนิ scratches ระดับความเสี่ยงต่ำ แสดงในตารางผล"],
        ["TC-03", "ตรวจตำหนิร้ายแรง", "ภาพมี crack หรือ crazing ชัดเจน",
         "พบกรอบตำหนิ ระดับความเสี่ยงสูง แสดงเน้นในสรุปผล"],
        ["TC-04", "ภาพไม่พบบริเวณโลหะ", "ภาพระยะใกล้ที่ Stage 1 หาโลหะไม่พบ",
         "ระบบใช้ fallback ตรวจทั้งภาพ และยังตรวจตำหนิได้"],
        ["TC-05", "ภาพหลายบริเวณ + ตำหนิซ้ำขอบ", "ภาพที่มีตำหนิคาบเกี่ยว 2 บริเวณ",
         "Cross-region NMS ตัดกรอบซ้ำ เหลือ 1 กรอบต่อ 1 ตำหนิ"],
        ["TC-06", "รูปแบบภาพหลากหลาย", "ภาพ grayscale / RGBA / .webp",
         "ระบบแปลงเป็น BGR และประมวลผลได้ตามปกติ"],
        ["TC-07", "โหมด --folder (CLI)", "โฟลเดอร์ภาพหลายไฟล์",
         "ได้ *_result.jpg / *_result.json ต่อภาพ และ _index.json สรุปทั้งชุด"],
        ["TC-08", "ภาพถ่ายจริง (domain gap)", "ภาพจาก real_test/",
         "ระบบรายงานผลได้ และตัวเลขสะท้อนช่องว่างเชิงโดเมนตามที่บันทึกไว้"],
    ], "กรณีทดสอบระบบ (Test Cases)", col_w=[1.4, 3.4, 4.4, 5.5])

    # ---------------- 9. การประเมินประสิทธิภาพ ----------------
    D.h("9. การประเมินประสิทธิภาพของระบบ", 2)
    D.table(["ด้านที่วัด", "ตัวชี้วัด", "วิธี/เครื่องมือ"], [
        ["ความแม่นยำของแบบจำลอง", "mAP@0.5, mAP@0.5–0.95 รายคลาสและภาพรวม",
         "evaluate.py --mode stage2 บนชุดทดสอบ"],
        ["ความครอบคลุม", "Precision, Recall, macro-F1 รายคลาส",
         "evaluate.py / tune_thresholds.py (จาก F1-vs-confidence บน validation)"],
        ["ประสิทธิภาพ Stage 1", "fallback rate, coverage, gt_area_kept, เวลา/ภาพ",
         "evaluate_stage1.py บนชุดทดสอบเต็ม"],
        ["การใช้งานจริง", "micro/macro P·R·F1 (pipeline vs baseline), เวลา/ภาพ",
         "evaluate_real.py บนภาพถ่ายจริง real_test/"],
        ["ความพึงพอใจผู้ใช้", "ค่าเฉลี่ยรายข้อ (Likert 5 ระดับ)",
         "แบบสอบถามหลังทดลองใช้หน้าเว็บ"],
    ], "ตัวชี้วัดและวิธีการประเมินประสิทธิภาพของระบบ", col_w=[3.4, 5.6, 5.7])

    # ---------------- 10. สถิติที่ใช้ ----------------
    D.h("10. สถิติที่ใช้ในการวิเคราะห์ข้อมูล", 2)
    D.bullets([
        "ค่าเฉลี่ยเลขคณิต (Mean) และส่วนเบี่ยงเบนมาตรฐาน (Standard Deviation) สำหรับสรุปผลการวัด "
        "ประสิทธิภาพแบบจำลอง (รายงานเป็น mean ± SD จากการฝึกซ้ำหลาย seed) และคะแนนความพึงพอใจรายข้อ",
        "ร้อยละ (Percentage) สำหรับสรุปสัดส่วน เช่น fallback rate และการกระจายของคลาส",
        "การแปลผลค่าเฉลี่ยความพึงพอใจตามเกณฑ์ 5 ระดับ (ตาราง 2)",
    ])


def bibliography(D):
    d = D.d
    d.add_page_break()
    hb = d.add_heading("บรรณานุกรม", level=1)
    hb.alignment = WD_ALIGN_PARAGRAPH.CENTER
    refs = [
        "Deep Vision Systems. (2025). How AI visual inspection transforms quality control in 2025.",
        "El Fassi, T. (2025). Optimizing quality control with AI-powered machine vision in manufacturing "
        "[Master’s thesis, Metropolia University of Applied Sciences]. Theseus.",
        "Huber, C., Knoll, D., & Guthe, M. (2025). Fully-synthetic training for visual quality inspection "
        "in automotive production. arXiv preprint.",
        "Jocher, G., Chaurasia, A., & Qiu, J. (2024). Ultralytics YOLO11. "
        "https://github.com/ultralytics/ultralytics",
        "Lee, C., Kim, Y., & Kim, H. (2024). Computer-vision-based product quality inspection and novel "
        "counting system. Applied System Innovation, 7(6), 127.",
        "Luo, Y., Du, Y., Wang, Z., Mo, J., Yu, W., & Dou, S. (2025). DScanNet: Packaging defect detection "
        "algorithm based on selective state space models. Algorithms, 18(1), 15–30.",
        "Mao, W. L., Wang, C. C., Chou, P. H., & Liu, Y. T. (2025). Automated defect detection for "
        "mass-produced electronic components based on YOLO object detection models. arXiv preprint.",
        "Mei, R. S., Jia, S., Li, G., Lee, S. Y., Musser, B., Keller, W., & Shao, C. (2025). Hybrid "
        "synthetic data generation with domain randomization for vision-based part inspection. arXiv preprint.",
        "Schmitt, R. H., & Stief, J. (2024). Production-ready end-to-end visual quality inspection for "
        "defect detection on surfaces based on a multi-stage AI system. Proceedings of DATA 2024, 145–152.",
        "Song, K., & Yan, Y. (2013). A noise robust method based on completed local binary patterns for "
        "hot-rolled steel strip surface defects. Applied Surface Science, 285, 858–864.",
        "Upchurch, P., & Niu, R. (2022). A dense material segmentation dataset for indoor and outdoor "
        "scene parsing. European Conference on Computer Vision (ECCV).",
        "Wang, H., Yang, X., Zhou, B., Shi, Z., Zhan, D., Huang, R., & Long, D. (2023). Strip surface "
        "defect detection algorithm based on YOLOv5. Materials, 16(15), 5344.",
        "Yadav, S., et al. (2024). Automated product defect detection using image processing techniques "
        "and YOLOv5. IJISRT, 9(6), 1120–1127.",
    ]
    for i, r in enumerate(refs, 1):
        par = d.add_paragraph(f"[{i}]  {r}")
        par.paragraph_format.left_indent = Cm(1.0)
        par.paragraph_format.first_line_indent = Cm(-1.0)
        par.paragraph_format.space_after = Pt(4)
        for rr in par.runs:
            rr.font.name = "TH Sarabun New"
            rr.font.size = Pt(14)
