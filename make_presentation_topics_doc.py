"""
สร้างไฟล์หัวข้อนำเสนอ (แบบกระชับ สำหรับก็อปไปทำสไลด์ Canva) -> docs/presentation_topics.docx

    python make_presentation_topics_doc.py

ครอบคลุมบทที่ 1 (1.1-1.8), บทที่ 2 (2.1-2.3), บทที่ 3 (การออกแบบระบบ) ตามสารบัญเล่มจริง
เนื้อหาสั้น เป็นวลี/bullet พร้อมขึ้นสไลด์ ไม่ใช่ย่อหน้ายาว — จุดที่ต้นฉบับเขียนเกินจริงถูกแก้ให้ตรง
กับ thesis_notes.md / NEXT_STEPS.md / results/*.json ของจริงแล้ว (มีเครื่องหมาย ⚠ กำกับ)

ต้องมี: python-docx
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor

BASE = Path(__file__).resolve().parent
DOCX = BASE / "docs" / "presentation_topics.docx"
FONT_TH = "TH Sarabun New"


def set_font(run, size=13, bold=False, color=None):
    run.font.name = FONT_TH
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def h1(doc, text):
    p = doc.add_paragraph()
    set_font(p.add_run(text), 20, bold=True)


def h2(doc, text):
    p = doc.add_paragraph()
    set_font(p.add_run(text), 16, bold=True, color=(0x1F, 0x4E, 0x79))
    p.space_before = Pt(12)


def h3(doc, text):
    p = doc.add_paragraph()
    set_font(p.add_run(text), 14, bold=True, color=(0x2E, 0x74, 0xB5))
    p.space_before = Pt(6)


def warn(doc, text):
    p = doc.add_paragraph()
    set_font(p.add_run("⚠ " + text), 11, bold=True, color=(0xB0, 0x00, 0x00))


def bullet(doc, text, size=12):
    p = doc.add_paragraph(style="List Bullet")
    set_font(p.add_run(text), size)


def para(doc, text, size=12, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, size)
    r.italic = italic


def table_simple(doc, rows):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            p = t.cell(i, j).paragraphs[0]
            set_font(p.add_run(cell), 11, bold=(i == 0))
    return t


def build():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = FONT_TH
    style.font.size = Pt(12)

    h1(doc, "หัวข้อนำเสนอโปรเจค (สรุปกระชับ — สำหรับทำสไลด์ Canva)")
    para(doc, "ระบบตรวจจับตำหนิผิวเหล็กด้วยปัญญาประดิษฐ์และอธิบายได้ (Explainable Steel Surface Defect Detection)", italic=True)
    para(doc, "⚠ = แก้จากต้นฉบับให้ตรงของจริงในโค้ด/ผลทดลอง", size=10, italic=True)

    # ============================================================ บทที่ 1
    h2(doc, "บทที่ 1 — บทนำ")

    h3(doc, "1.1 ความเป็นมาและความสำคัญของปัญหา")
    bullet(doc, "ตรวจสอบผิวเหล็กด้วยสายตา → ช้า + ไม่สม่ำเสมอ (human fatigue)")
    bullet(doc, "AI ที่มีอยู่ส่วนใหญ่เป็น \"กล่องดำ\" — บอกผล ไม่บอกเหตุผล")
    bullet(doc, "พบตำหนิหลุดรอดถึงผู้บริโภคจริง (รอยขีดข่วน/รอยแตกบนเหล็กเส้น/แผงเหล็ก)")

    h3(doc, "1.2 วัตถุประสงค์")
    bullet(doc, "พัฒนาต้นแบบตรวจจับตำหนิผิวเหล็กแบบ 2-Stage รองรับภาพนิ่ง/ถ่ายภาพ/เรียลไทม์ (กล้อง-หน้าจอสด)")
    bullet(doc, "แสดงตำแหน่ง/ประเภท/ค่าความเชื่อมั่น/ระดับความเสี่ยง ให้ผู้ใช้ตีความผลเอง")
    warn(doc, "ตัด \"Dashboard/Alert System, Pass/Fail อัตโนมัติ\" — ไม่มีจริง (real-time live scan มีจริง ไม่ตัด)")

    h3(doc, "1.3 แนวคิดและหลักการ")
    bullet(doc, "Detection: ภาพ → โมเดล → bbox + class + confidence")
    bullet(doc, "Explainability: Defect Ratio → จัดระดับความเสี่ยง (ต่ำ/กลาง/สูง)")
    warn(doc, "ระบบให้ข้อมูลประกอบการตัดสินใจ ไม่ได้ฟันธง Pass/Fail เอง")

    h3(doc, "1.4 ขอบเขต")
    bullet(doc, "prototype ทำงานบนเครื่องเดียว ผ่านเว็บ Gradio")
    bullet(doc, "รองรับ 3 ช่องทางนำเข้าภาพ: อัปโหลด / ถ่ายภาพ / เรียลไทม์ (สแกนกล้อง-หน้าจอสด ผ่าน getDisplayMedia)")
    bullet(doc, "ตรวจ 8 คลาส: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches, Rust, Crack")
    bullet(doc, "อธิบายผลด้วย bbox / Defect Ratio / Confidence / Severity")
    warn(doc, "ตัดเฉพาะ \"สายพานลำเลียงในโรงงาน + ต่อ PLC\" — ไม่มีจริง / แต่ real-time live scan (กล้อง/หน้าจอ) มีจริง เก็บไว้เป็นจุดขาย")

    h3(doc, "1.5 นิยามศัพท์เฉพาะ")
    bullet(doc, "YOLO — ตรวจตำแหน่ง+ประเภทวัตถุพร้อมกัน (ใช้ YOLO11)")
    bullet(doc, "Bounding Box / Confidence Score — ตำแหน่ง / ความมั่นใจของผลตรวจจับ")
    bullet(doc, "Defect Ratio / Severity Level — สัดส่วนพื้นที่ตำหนิ / ระดับความรุนแรง")
    bullet(doc, "Dataset / Roboflow — ชุดข้อมูลภาพ / เครื่องมือ annotate (เทรนโมเดลเองด้วย Ultralytics YOLO)")

    h3(doc, "1.6 สถานะงาน (แทนตารางแผนเดิม)")
    warn(doc, "แผนเดิมเขียนราวกับยังไม่เริ่ม — ใช้ตารางนี้แทน")
    table_simple(doc, [
        ["ช่วง", "สถานะ", "งาน"],
        ["ก่อน ก.ย. 69", "✅", "รวม dataset + แก้ label + grayscale"],
        ["ต้น ก.ย. 69", "✅", "แก้ leakage, multi-seed mAP50 0.867±0.010"],
        ["กลาง ก.ย. 69", "✅", "Stage 1 ablation, rust จริง 8/12→12/12"],
        ["13 ก.ย. 69", "✅", "ทดลอง scratches/crack เพิ่ม → negative ablation"],
        ["ตอนนี้", "🔴", "เคลียร์ baseline กับอาจารย์"],
    ])

    h3(doc, "1.7 เครื่องมือ")
    bullet(doc, "Python 3.11, Ultralytics YOLO11, PyTorch, OpenCV")
    bullet(doc, "DMS46 (Stage 1 หาบริเวณโลหะ), Gradio (UI)")
    bullet(doc, "GPU: RTX 3050 6GB · เก็บผลเป็นไฟล์ภาพ+JSON")
    warn(doc, "ตัด SQLite/PostgreSQL/Streamlit/React — ไม่ได้ใช้จริง")

    h3(doc, "1.8 ประโยชน์ที่คาดว่าจะได้รับ")
    bullet(doc, "ผู้ใช้: ตรวจเร็วขึ้น เห็นเหตุผลประกอบ ตรวจย้อนหลังได้")
    bullet(doc, "กระบวนการ QC: มาตรฐานสม่ำเสมอกว่าคนตรวจเอง")
    bullet(doc, "AI: ได้ pipeline 2-Stage + ชุดข้อมูล 8 คลาส ต้นแบบ")
    bullet(doc, "ผู้จัดทำ: ประสบการณ์ครบวงจร data → train → design → เขียนรายงาน")

    # ============================================================ บทที่ 2
    h2(doc, "บทที่ 2 — เอกสารและงานวิจัยที่เกี่ยวข้อง")

    h3(doc, "2.1 ทฤษฎีที่เกี่ยวข้อง")
    bullet(doc, "Deep Learning/CNN — เรียนรู้ลักษณะตำหนิจากภาพอัตโนมัติ")
    bullet(doc, "Image Segmentation — DMS46 หาบริเวณ Metal ก่อนตรวจจับ")
    bullet(doc, "Object Detection/YOLO — ตรวจตำแหน่ง+ประเภทในกระบวนการเดียว")
    bullet(doc, "Precision/Recall/mAP/IoU — ตัวชี้วัดหลักของงาน")
    bullet(doc, "Two-Stage Pipeline — แยกหาบริเวณวัสดุ กับตรวจจับตำหนิ")
    warn(doc, "XAI: พูดแค่ bbox/ratio/confidence — ไม่ใช้ LIME/SHAP/Grad-CAM จริง")

    h3(doc, "2.2 งานวิจัยที่เกี่ยวข้อง")
    bullet(doc, "NEU-DET — พื้นฐาน dataset + 6 ใน 8 คลาส")
    bullet(doc, "YOLO สำหรับ steel defect — ยืนยันแนวทางที่เลือกใช้")
    bullet(doc, "Explainable Hybrid AI / NEU-XAI — แนวคิดแยกตรวจจับ+อธิบายผล (เราเรียบง่ายกว่า)")
    bullet(doc, "ช่องว่างที่เติมเต็ม: แยกบริเวณวัสดุก่อนตรวจจับ + จัดการกรอบซ้ำ (Cross-Region NMS) + อธิบายผลจากข้อมูลจริง")

    h3(doc, "2.3 รีวิวระบบที่เกี่ยวข้อง")
    bullet(doc, "กลุ่ม YOLO ล้วน — เร็ว แต่ไม่แยกบริเวณ ไม่อธิบายผล")
    bullet(doc, "กลุ่มมี UI — ใช้ง่ายขึ้น แต่ไม่มีคำอธิบายเพิ่ม")
    bullet(doc, "กลุ่ม Explainable AI — อธิบายลึกแต่มักเป็นระดับวิจัย ไม่ใช่ระบบใช้งานจริง")
    bullet(doc, "จุดยืนของเรา: DMS46 + YOLO11 + Cross-Region NMS + อธิบายผลเรียบง่าย = ต้นแบบใช้งานได้จริง")

    # ============================================================ บทที่ 3
    h2(doc, "บทที่ 3 — วิธีดำเนินงานและการออกแบบระบบ (ของจริง 100% ตรงกับโค้ด)")

    h3(doc, "3.1 ประชากร/กลุ่มตัวอย่าง")
    bullet(doc, "ภาพผิวเหล็ก: NEU-DET 6 คลาส + rust + crack เพิ่มเอง")
    bullet(doc, "แบ่ง train/val/test แบบ group-aware กัน data leakage")
    bullet(doc, "เกณฑ์คัดออก: ภาพเบลอ/มืด/ซ้ำ/ระบุประเภทไม่ได้")

    h3(doc, "3.4 การวิเคราะห์ความต้องการ (สรุป)")
    bullet(doc, "Functional หลัก: รับภาพ → หาบริเวณโลหะ → ตรวจจับ 8 คลาส → กรอง confidence → แสดงผล")
    bullet(doc, "Nonfunctional หลัก: รันบนเครื่องเดียว (local), ไม่ต้องต่อ Cloud, ปรับ threshold ผ่านไฟล์ config ได้")

    h3(doc, "3.5 การออกแบบระบบ")
    bullet(doc, "System Architecture: ภาพ → Stage 1 (DMS46 หา Metal Mask) → สร้าง Region "
                "→ Stage 2 (YOLO11 ตรวจจับ) → Cross-Region NMS → กรอง per-class threshold → ผลลัพธ์")
    bullet(doc, "Fallback: ถ้าไม่พบบริเวณโลหะ/สัดส่วนต่ำเกิน → ใช้ทั้งภาพเป็น Region แทน")
    bullet(doc, "Use Case หลัก 10 ตัว: อัปโหลดภาพ, ปรับความไว, ตรวจจับ 2-Stage, แสดงผล, บันทึกผล, "
                "เตรียมข้อมูล, เทรน Stage1/Stage2, ปรับ threshold, ทดสอบโมเดล")
    bullet(doc, "Database Design: ไม่ใช้ DBMS — เก็บเป็นไฟล์ (ภาพนำเข้า/ผลลัพธ์, JSON ผลตรวจจับ, thresholds.json, ไฟล์โมเดล)")
    bullet(doc, "UI Design (Gradio): อัปโหลดภาพ → เลือกโหมดความไว (มาตรฐาน/ไว/ไวมาก) → ผลลัพธ์พร้อมกรอบ+ระดับความเสี่ยง")

    # ============================================================ ผลจริง + ข้อจำกัด
    h2(doc, "ผลการทดลอง (จุดขายหลัก — ตรงของจริง)")
    table_simple(doc, [
        ["ประเด็น", "ผล"],
        ["mAP@0.5 (multi-seed n=4)", "0.867 ± 0.010"],
        ["คลาสอ่อนสุด", "crack (~0.687)"],
        ["Stage 1 ablation", "negative — ตัดออกได้ตามผลจริง"],
        ["rust บนภาพถ่ายจริง", "8/12 → 12/12 (3 รอบเก็บข้อมูล)"],
        ["scratches/crack เพิ่มจากแหล่งเปิด", "negative/inconclusive — ต้องใช้ภาพ scene จริงเท่านั้น"],
    ])

    h2(doc, "ข้อจำกัดที่ต้องพูดเอง")
    bullet(doc, "prototype ไม่ใช่ระบบใช้งานจริง / ไม่ real-time / ไม่มี DB")
    bullet(doc, "อธิบายผลแบบเรียบง่าย ไม่ใช่ LIME/SHAP")
    bullet(doc, "6 ใน 8 คลาส ยังตรวจภาพจริงนอกแล็บได้จำกัด — rust คลาสเดียวที่พิสูจน์แล้วว่าใช้งานได้จริง")

    Path(BASE / "docs").mkdir(exist_ok=True)
    doc.save(DOCX)
    print(f"เขียนแล้ว: {DOCX}")


if __name__ == "__main__":
    build()
