# -*- coding: utf-8 -*-
"""
make_ui_doc.py — เอกสารแยก "6.9 การออกแบบส่วนติดต่อผู้ใช้ (User Interface Design)"
พร้อมตัวอย่างการใช้งานจริง 10 ตัวอย่าง (ผลตรวจแต่ละชนิดตำหนิ + สถานะต่าง ๆ) และคำอธิบาย

    python make_ui_doc.py   ->  docs/6.9_การออกแบบส่วนติดต่อผู้ใช้.docx

ต้องสร้างรูปก่อน (ทำอัตโนมัติถ้ายังไม่มี): เรนเดอร์ผลตรวจจาก app._analyze ลง figures/thesis/ui_c_*.jpg
"""
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures" / "thesis"
OUT = BASE / "docs" / "6.9_การออกแบบส่วนติดต่อผู้ใช้.docx"
TH = "TH Sarabun New"


# ---------------------------------------------------------------- render figures
def ensure_figs():
    need = ["ui_c_crazing", "ui_c_inclusion", "ui_c_patches", "ui_c_pitted",
            "ui_c_rolled", "ui_c_scratches", "ui_c_rust", "ui_c_crack",
            "ui_c_rust_real", "ui_c_none", "ui_res_stage1"]
    if all((FIG / f"{n}.jpg").exists() for n in need):
        return
    import cv2
    import app as A

    class _P:
        def __call__(self, *a, **k):
            pass

    prog = _P()
    mk = list(A._available_models())[0]
    jobs = [
        ("test_images/crazing_example.jpg", "ไว", "ui_c_crazing"),
        ("test_images/inclusion_example.jpg", "ไว", "ui_c_inclusion"),
        ("test_images/patches_example.jpg", "มาตรฐาน", "ui_c_patches"),
        ("test_images/pitted_surface_example.jpg", "ไว", "ui_c_pitted"),
        ("test_images/rolled-in_scale_example.jpg", "ไว", "ui_c_rolled"),
        ("test_images/scratches_example.jpg", "มาตรฐาน", "ui_c_scratches"),
        ("test_images/rust_example.jpg", "มาตรฐาน", "ui_c_rust"),
        ("test_images/crack_example.jpg", "มาตรฐาน", "ui_c_crack"),
        ("real_test/images/real_003_rebar_stack_rust.jpg", "ไว", "ui_c_rust_real"),
        ("test_images/normal_steel_example.jpg", "มาตรฐาน", "ui_c_none"),
    ]
    for src, sens, tag in jobs:
        im = cv2.imread(str(BASE / src))
        if im is None:
            continue
        h, w = im.shape[:2]
        if max(h, w) < 1100:
            s = 1100 / max(h, w)
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        a, s1, *_ = A._analyze(rgb, 0.4, False, sens, mk, True, prog)
        cv2.imwrite(str(FIG / f"{tag}.jpg"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        if tag == "ui_c_rust_real" and s1 is not None:
            cv2.imwrite(str(FIG / "ui_res_stage1.jpg"),
                        cv2.cvtColor(s1, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- docx helpers
def _rf(run):
    run.font.name = TH
    rpr = run._element.get_or_add_rPr()
    rf = OxmlElement("w:rFonts"); rf.set(qn("w:cs"), TH); rpr.append(rf)


class D:
    def __init__(self):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = TH; st.font.size = Pt(16)
        st.element.rPr.rFonts.set(qn("w:cs"), TH)
        for i in range(1, 4):
            hs = self.d.styles[f"Heading {i}"]
            hs.font.name = TH; hs.font.size = Pt(20 - i * 2)
            hs.font.color.rgb = RGBColor(0, 0, 0); hs.font.bold = True
            hs.element.rPr.rFonts.set(qn("w:cs"), TH)
        for s in self.d.sections:
            s.left_margin = Cm(3.0); s.right_margin = Cm(2.0)
            s.top_margin = s.bottom_margin = Cm(2.5)
        self.fig_n = 0

    def h(self, t, lvl=2):
        self.d.add_heading(t, level=lvl)

    def p(self, t, indent=True):
        par = self.d.add_paragraph(t)
        par.paragraph_format.first_line_indent = Cm(1.27 if indent else 0)
        par.paragraph_format.space_after = Pt(6)
        par.paragraph_format.line_spacing = 1.15
        for r in par.runs:
            _rf(r)
        return par

    def kv(self, key, val):
        par = self.d.add_paragraph()
        r1 = par.add_run(f"{key}  "); r1.bold = True; _rf(r1); r1.font.size = Pt(16)
        r2 = par.add_run(val); _rf(r2); r2.font.size = Pt(16)
        par.paragraph_format.space_after = Pt(3)
        par.paragraph_format.left_indent = Cm(0.6)

    def bullets(self, items):
        for it in items:
            par = self.d.add_paragraph(it, style="List Bullet")
            par.paragraph_format.space_after = Pt(2)
            for r in par.runs:
                _rf(r); r.font.size = Pt(16)

    def figure(self, name, caption, width=5.4):
        self.fig_n += 1
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.add_run().add_picture(str(FIG / name), width=Inches(width))
        cap = self.d.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(f"ภาพ 6.9-{self.fig_n}  {caption}")
        _rf(r); r.font.size = Pt(14)
        cap.paragraph_format.space_after = Pt(14)

    def table(self, headers, rows, col_w=None):
        t = self.d.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"
        for i, hc in enumerate(headers):
            t.rows[0].cells[i].text = str(hc)
            for pr in t.rows[0].cells[i].paragraphs:
                for r in pr.runs:
                    _rf(r); r.font.size = Pt(13); r.font.bold = True
        for row in rows:
            cs = t.add_row().cells
            for i, v in enumerate(row):
                cs[i].text = str(v)
                for pr in cs[i].paragraphs:
                    for r in pr.runs:
                        _rf(r); r.font.size = Pt(13)
        if col_w:
            for i, w in enumerate(col_w):
                for row in t.rows:
                    row.cells[i].width = Cm(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(6)

    def save(self):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        self.d.save(str(OUT))
        print("เขียน:", OUT.relative_to(BASE))


# ---------------------------------------------------------------- content
EXAMPLES = [
    ("ui_c_rust.jpg", "ผลการตรวจ — สนิม (rust) บนผิวเหล็ก",
     "แสดงการตรวจพบตำหนิความเสี่ยง “สูง” — กรณีที่ระบบต้องเน้นให้ผู้ใช้เห็นชัดที่สุด",
     ["ภาพผลลัพธ์: กรอบสีแดงล้อมบริเวณสนิม พร้อมป้ายภาษาไทย “สนิม” และค่าความมั่นใจกำกับบนภาพ",
      "การ์ดสรุปผลด้านขวา: แถบสีแดง หัวข้อ “พบตำหนิความเสี่ยงสูง” ระบุชนิดและจำนวน",
      "ตารางรายการตำหนิ: 1 แถว — บริเวณ / สนิม / rust / % ความมั่นใจ / ระดับความเสี่ยง “สูง”"],
     "สนิมและรอยแตกร้าวถูกกำหนดให้เป็นความเสี่ยงสูง ระบบจึงใช้สีแดงทั้งกรอบบนภาพและแถบการ์ด "
     "เพื่อให้ผู้ตรวจสอบสังเกตได้ทันทีโดยไม่ต้องอ่านรายละเอียด"),

    ("ui_c_crack.jpg", "ผลการตรวจ — รอยแตกร้าว (crack)",
     "ตำหนิความเสี่ยง “สูง” อีกชนิด แสดงรูปแบบการนำเสนอที่สอดคล้องกับกรณีสนิม",
     ["กรอบสีแดง + ป้าย “รอยแตกร้าว (xx%)” บนภาพ",
      "การ์ดสรุปผลสีแดง หัวข้อ “พบตำหนิความเสี่ยงสูง: รอยแตกร้าว”",
      "ตารางระบุคลาสภาษาอังกฤษ (crack) กำกับไว้ให้ตรวจสอบย้อนกลับได้"],
     "ใช้รูปแบบสี/ตำแหน่งเดียวกับทุกชนิดตำหนิ เพื่อให้ผู้ใช้เรียนรู้การอ่านผลเพียงครั้งเดียว "
     "แล้วใช้ได้กับทุกกรณี (consistency)"),

    ("ui_c_crazing.jpg", "ผลการตรวจ — รอยแตกลายงา (crazing)",
     "ตำหนิระดับ “ปานกลาง–สูง” — แสดงว่าระบบจัดลำดับความเสี่ยงได้ละเอียดกว่าแค่ผ่าน/ไม่ผ่าน",
     ["กรอบตำหนิ + ป้ายไทยบนภาพ",
      "การ์ดสรุปผล: หัวข้อระบุชนิดตำหนิและจำนวนชนิดที่พบ",
      "ตาราง: คอลัมน์ความเสี่ยงแสดง “ปานกลาง-สูง”"],
     "ระดับความเสี่ยงมาจากตารางกำหนดไว้ล่วงหน้าต่อชนิดตำหนิ (Severity by class) ช่วยให้ผู้ใช้ "
     "ตัดสินใจจัดลำดับงานตรวจซ้ำได้"),

    ("ui_c_inclusion.jpg", "ผลการตรวจ — สิ่งแปลกปลอมฝังใน (inclusion)",
     "ตำหนิความเสี่ยง “ปานกลาง” — การ์ดสรุปผลเปลี่ยนเป็นโทนส้ม/เหลืองแทนแดง",
     ["กรอบตำหนิ + ป้ายไทยบนภาพ",
      "การ์ดสรุปผล: แถบสีส้ม หัวข้อ “พบตำหนิ N ชนิด” พร้อมข้อความ “ไม่มีชนิดที่จัดเป็นความเสี่ยงสูง”",
      "ตาราง: ระดับความเสี่ยง “ปานกลาง”"],
     "แยกสีของการ์ดตามระดับความเสี่ยงสูงสุดที่พบ (แดง = มีความเสี่ยงสูง, ส้ม = พบตำหนิแต่ไม่สูง) "
     "เพื่อให้ผู้ใช้กะระดับความเร่งด่วนจากสีได้"),

    ("ui_c_patches.jpg", "ผลการตรวจ — รอยแผ่น/ผิวลอก (patches) หลายจุด",
     "กรณีพบตำหนิหลายจุดในภาพเดียว — แสดงการรวมผลและการจัดตาราง",
     ["ภาพผลลัพธ์: กรอบตำหนิหลายกรอบ",
      "ตารางรายการตำหนิ: หลายแถว เรียงตามระดับความเสี่ยงจากมากไปน้อย แล้วตามด้วยค่าความมั่นใจ",
      "การ์ดสรุปผล: สรุปจำนวน “ชนิด” ที่พบ (ไม่ใช่จำนวนกรอบ) เพื่อไม่ให้ตัวเลขน่าตกใจเกินจริง"],
     "ระบบทำ Cross-region NMS รวมกรอบซ้ำข้ามบริเวณก่อนแสดง เพื่อไม่ให้ผู้ใช้เห็นกรอบซ้อนของ "
     "ตำหนิเดียวกัน"),

    ("ui_c_pitted.jpg", "ผลการตรวจ — ผิวขรุขระเป็นหลุม (pitted surface)",
     "แสดงผลบนตำหนิที่ขอบเขตไม่ชัด — ระบบยังตีกรอบบริเวณที่เป็นหลุมให้เห็น",
     ["กรอบตำหนิครอบบริเวณผิวเป็นหลุม + ป้ายไทย",
      "ตาราง: ระดับความเสี่ยง “ปานกลาง”",
      "ส่วน “รายละเอียดทางเทคนิค” (พับไว้) บอกจำนวนบริเวณและสัดส่วนพื้นที่เหล็กที่ Stage 1 พบ"],
     "ความหนาของกรอบและขนาดฟอนต์ป้ายปรับตามขนาดภาพอัตโนมัติ ทำให้อ่านออกทั้งบนภาพเล็กและภาพใหญ่"),

    ("ui_c_rolled.jpg", "ผลการตรวจ — สะเก็ดฝังจากการรีด (rolled-in scale)",
     "ครบทั้ง 6 ชนิดตำหนิจากชุด NEU — ยืนยันว่าหน้าจอเดียวรองรับทุกคลาส",
     ["กรอบตำหนิ + ป้ายไทย “สะเก็ดฝังจากการรีด”",
      "ตาราง: คลาส rolled-in_scale, ระดับ “ปานกลาง”"],
     "ป้ายภาษาไทยทุกชนิดกำหนดไว้ในโค้ด (DEFECT_INFO) ผู้ใช้ที่ไม่คุ้นศัพท์อังกฤษก็อ่านผลได้"),

    ("ui_c_scratches.jpg", "ผลการตรวจ — รอยขีดข่วน (scratches)",
     "ตำหนิความเสี่ยง “ต่ำ” — การ์ดสรุปผลยังเป็นโทนส้ม แต่ข้อความชี้ว่าไม่ใช่ความเสี่ยงสูง",
     ["กรอบตำหนิ + ป้ายไทย “รอยขีดข่วน”",
      "การ์ดสรุปผล: “พบตำหนิ 1 ชนิด … ไม่มีชนิดที่จัดเป็นความเสี่ยงสูง”",
      "ตาราง: ระดับความเสี่ยง “ต่ำ”"],
     "แม้ความเสี่ยงต่ำก็ยังแสดงผลครบ เพื่อให้ผู้ใช้ตัดสินใจเองว่าจะปล่อยผ่านหรือไม่ "
     "(ระบบเป็นผู้ช่วยคัดกรอง ไม่ตัดสินแทน)"),

    ("ui_c_rust_real.jpg", "ผลการตรวจ — ภาพถ่ายจริงระดับสถานที่ (เหล็กเส้นสนิม)",
     "กรณีภาพถ่ายจริงมุมกว้าง พื้นหลังรก — แสดงว่าหน้าจอเดียวใช้ได้ทั้งภาพแล็บและภาพถ่ายจริง",
     ["ภาพผลลัพธ์: กรอบสนิมหลายจุดบนเหล็กเส้นจริง",
      "ระบบเพิ่มรอบ “ตรวจทั้งภาพ” อัตโนมัติเมื่อ Stage 1 แตกเหล็กเป็นหลายบริเวณ",
      "ภาพ Stage 1 ด้านล่างแสดงบริเวณที่ระบบพิจารณาว่าเป็นเหล็ก"],
     "ช่องอัปโหลดรองรับทั้งลากไฟล์ วางจากคลิปบอร์ด และเลือกจากแกลเลอรีตัวอย่าง เพื่อให้ผู้มา "
     "ทดลอง (เช่น ถ่ายรูปเหล็กแล้วอัปขึ้น) ใช้ได้ทันที"),

    ("ui_c_none.jpg", "ผลการตรวจ — ไม่พบตำหนิ (ผิวเหล็กปกติ)",
     "สถานะ “ผ่าน” — การ์ดสรุปผลเปลี่ยนเป็นโทนเขียว ตารางว่าง",
     ["ภาพผลลัพธ์: ไม่มีกรอบตำหนิ",
      "การ์ดสรุปผล: แถบสีเขียว หัวข้อ “ไม่พบตำหนิพื้นผิว”",
      "ตารางรายการตำหนิ: ไม่มีแถว"],
     "ให้ feedback ที่ชัดเจนแม้ในกรณีไม่พบอะไร ผู้ใช้จึงมั่นใจว่าระบบทำงานแล้ว ไม่ใช่ค้าง"),
]


def main():
    ensure_figs()
    doc = D()
    d = doc.d

    t = d.add_heading("6.9 การออกแบบส่วนติดต่อผู้ใช้ (User Interface Design)", level=1)
    t.runs[0].font.size = Pt(20)

    doc.p("ส่วนติดต่อผู้ใช้ของระบบพัฒนาด้วยไลบรารี Gradio ในรูปแบบเว็บแอปพลิเคชันหน้าเดียว "
          "(Single-page) เปิดใช้งานผ่านเบราว์เซอร์ที่ http://127.0.0.1:7860 โดยตั้งเป้าหมายการ "
          "ออกแบบไว้ 4 ข้อ")
    doc.bullets([
        "ใช้งานง่าย ขั้นตอนเดียว: อัปโหลดหรือวางภาพ แล้วระบบตรวจให้อัตโนมัติ ไม่ต้องตั้งค่าก่อน",
        "อธิบายผลได้ (Explainable): แสดงกรอบตำแหน่งตำหนิ ป้ายชนิดภาษาไทย ค่าความมั่นใจ ระดับความ "
        "เสี่ยง และภาพบริเวณที่ระบบพิจารณา (Stage 1)",
        "เน้นความเสี่ยงสูง: ใช้สี (แดง/ส้ม/เขียว) ทั้งบนภาพและการ์ดสรุปผล ให้ผู้ใช้กะระดับความเร่งด่วน "
        "ได้จากสายตา",
        "รองรับทั้งคอมพิวเตอร์และมือถือ (Responsive): คอลัมน์ปรับเรียงตามความกว้างจอ ภาพย่อ/ขยาย "
        "ตามอัตราส่วนและขนาดจอ",
    ])

    doc.h("6.9.1 โครงหน้าจอโดยรวม (Layout)", 2)
    doc.p("หน้าจอแบ่งเป็น 4 ส่วนจากบนลงล่าง: (1) หัวเรื่องและคำอธิบายระบบ (2) แถวสองคอลัมน์ — "
          "ซ้ายเป็นส่วนรับภาพและตัวเลือก ขวาเป็นการ์ดสรุปผลและตารางรายการตำหนิ (3) ภาพผลลัพธ์เต็ม "
          "ความกว้าง (4) ภาพผลของ Stage 1 และส่วนรายละเอียดทางเทคนิค (พับไว้)")
    if (FIG / "th_wireframe.png").exists():
        doc.figure("th_wireframe.png", "โครงหน้าจอ (Wireframe) ของหน้าเว็บระบบ", width=6.4)

    doc.h("6.9.2 องค์ประกอบหลักของหน้าจอ", 2)
    doc.table(["องค์ประกอบ", "หน้าที่", "หมายเหตุการออกแบบ"], [
        ["ช่องรับภาพ (gr.Image)", "อัปโหลด / วางจากคลิปบอร์ด / ลากไฟล์",
         "ตรวจอัตโนมัติเมื่อภาพเปลี่ยน — ไม่ต้องกดปุ่ม"],
        ["ปุ่ม “ตรวจสอบ”", "สั่งประมวลผลซ้ำด้วยตัวเลือกปัจจุบัน", "ปุ่มหลัก ขนาดใหญ่ กดง่ายบนมือถือ"],
        ["ตัวเลือกขั้นสูง (Accordion)", "เลือกโมเดล / โหมดความไว / confidence / TTA / Stage 0",
         "พับไว้ — หน้าเริ่มต้นไม่มีปุ่มปรับให้สับสน"],
        ["แกลเลอรีภาพตัวอย่าง", "กดเพื่อลองตรวจทันที", "รวมภาพชุด benchmark และภาพถ่ายจริงไว้ด้วยกัน"],
        ["การ์ดสรุปผล (verdict)", "สรุปสถานะเป็นข้อความ + สี", "อ่านได้ใน 1 บรรทัด ไม่ต้องดูตาราง"],
        ["ตารางรายการตำหนิ (gr.Dataframe)", "รายละเอียดทุกจุด: บริเวณ ชนิด คลาส ความมั่นใจ ความเสี่ยง",
         "เรียงตามความเสี่ยง เลื่อนแนวนอนได้บนจอแคบ"],
        ["ภาพผลลัพธ์", "ภาพต้นฉบับ + กรอบ + ป้ายไทย", "ปรับขนาดตามอัตราส่วนภาพ จำกัดไม่ให้ล้นจอ"],
        ["ภาพ Stage 1", "บริเวณที่ระบบพิจารณาว่าเป็นเหล็ก (เขียว) / ตรวจทั้งภาพ (ส้ม)",
         "แสดงตลอด เพื่อความโปร่งใสของการทำงาน"],
    ], col_w=[3.6, 5.4, 5.5])

    doc.h("6.9.3 สถานะของการ์ดสรุปผล (Verdict States)", 2)
    doc.table(["สถานะ", "เงื่อนไข", "สี", "ข้อความหลัก"], [
        ["ความเสี่ยงสูง", "พบตำหนิที่จัดระดับ “สูง” หรือ “ปานกลาง-สูง”", "แดง", "พบตำหนิความเสี่ยงสูง: …"],
        ["พบตำหนิ", "พบตำหนิ แต่ไม่มีชนิดความเสี่ยงสูง", "ส้ม", "พบตำหนิ N ชนิด: …"],
        ["อาจมีตำหนิ", "มีเฉพาะผลความมั่นใจต่ำกว่าเกณฑ์ (tentative)", "เทา-ฟ้า",
         "อาจมีตำหนิ (ความมั่นใจต่ำ) — แนะนำให้ตรวจซ้ำ"],
        ["ไม่พบตำหนิ", "ไม่พบตำหนิใด ๆ", "เขียว", "ไม่พบตำหนิพื้นผิว"],
        ["ไม่พบพื้นผิวเหล็ก", "Stage 0 + Stage 1 + Stage 2 เห็นตรงกันว่าไม่ใช่เหล็ก", "เทา",
         "ไม่พบพื้นผิวเหล็กในภาพนี้"],
    ], col_w=[2.8, 5.6, 1.6, 4.5])

    doc.h("6.9.4 ตัวอย่างการใช้งานจริง 10 ตัวอย่าง", 2)
    doc.p("ตัวอย่างต่อไปนี้เป็นผลตรวจจริงจากระบบ (ยังไม่ตกแต่งเพิ่ม) ครอบคลุมตำหนิทั้ง 8 ชนิด "
          "ทั้งภาพชุดมาตรฐานและภาพถ่ายจริง รวมถึงกรณีไม่พบตำหนิ")
    for i, (img, title, purpose, elems, rationale) in enumerate(EXAMPLES, 1):
        doc.h(f"ตัวอย่างที่ {i} — {title}", 3)
        if (FIG / img).exists():
            doc.figure(img, title, width=5.2)
        doc.kv("จุดประสงค์:", purpose)
        p = d.add_paragraph(); r = p.add_run("องค์ประกอบ UI ที่เกี่ยวข้อง:")
        r.bold = True; _rf(r); r.font.size = Pt(16)
        p.paragraph_format.left_indent = Cm(0.6); p.paragraph_format.space_after = Pt(2)
        doc.bullets(elems)
        doc.kv("เหตุผลการออกแบบ:", rationale)
        d.add_paragraph().paragraph_format.space_after = Pt(4)

    doc.h("6.9.5 ภาพผลของ Stage 1 (ความโปร่งใสของการทำงาน)", 2)
    doc.p("ใต้ภาพผลลัพธ์ ระบบแสดงภาพของ Stage 1 เสมอ โดยระบายสีเขียวทับบริเวณที่ DMS46 ระบุว่าเป็น "
          "วัสดุโลหะ และใช้กรอบสีส้มเมื่อระบบตัดสินใจ “ตรวจทั้งภาพ” (fallback) ผู้ใช้จึงตรวจสอบได้ว่า "
          "ระบบพิจารณาพื้นที่ใดของภาพ")
    if (FIG / "ui_res_stage1.jpg").exists():
        doc.figure("ui_res_stage1.jpg", "ภาพผลของ Stage 1 — บริเวณที่ระบบพิจารณาว่าเป็นเหล็ก", width=5.4)

    doc.h("6.9.6 การรองรับคอมพิวเตอร์และมือถือ (Responsive Design)", 2)
    doc.bullets([
        "จอกว้าง (คอมพิวเตอร์): ส่วนรับภาพและส่วนสรุปผลวางเคียงกันสองคอลัมน์ ภาพผลลัพธ์กว้างเต็มพื้นที่ "
        "(สูงสุด ~72% ของความสูงจอ)",
        "จอแคบ (มือถือ/แท็บเล็ต): สองคอลัมน์เรียงต่อกันเป็นแนวตั้งอัตโนมัติ ตัวอักษรและระยะขอบย่อลง "
        "ภาพจำกัดความสูงที่ ~56% ของจอ ตารางเลื่อนแนวนอนได้",
        "ภาพทั้งหมดใช้ object-fit แบบ contain — ไม่ถูกครอบตัด ไม่ว่าจะเป็นภาพแนวตั้งหรือแนวนอน",
        "ปุ่มหลักขนาดใหญ่ (size = lg) เพื่อให้กดง่ายด้วยนิ้ว",
    ])

    doc.h("6.9.7 การไหลของเหตุการณ์ (Event Flow)", 2)
    doc.p("เมื่อผู้ใช้อัปโหลดภาพ เปลี่ยนโหมดความไว เปลี่ยนโมเดล หรือกดปุ่ม “ตรวจสอบ” ระบบจะเรียก "
          "ฟังก์ชัน analyze() ด้วยพารามิเตอร์ [ภาพ, confidence, ตรวจละเอียด, โหมดความไว, โมเดล, "
          "Stage 0] และคืนผลลัพธ์ 5 ส่วนพร้อมกัน ได้แก่ ภาพผลลัพธ์ / ภาพ Stage 1 / การ์ดสรุปผล / "
          "ตารางรายการตำหนิ / รายละเอียดทางเทคนิค")

    doc.h("6.9.8 หลักการออกแบบโดยสรุป", 2)
    doc.bullets([
        "Consistency — ทุกชนิดตำหนิใช้รูปแบบสี/ตำแหน่ง/ป้ายเดียวกัน ผู้ใช้เรียนรู้ครั้งเดียวใช้ได้ทั้งหมด",
        "Progressive disclosure — ค่าเริ่มต้นเรียบง่าย ตัวเลือกขั้นสูงและรายละเอียดทางเทคนิคพับซ่อนไว้",
        "Explainability first — ทุกผลลัพธ์มาพร้อมตำแหน่ง ค่าความมั่นใจ ระดับความเสี่ยง และภาพ Stage 1",
        "Feedback — มีสถานะชัดเจนทุกกรณี รวมถึงกรณีไม่พบตำหนิและกรณีภาพไม่ใช่เหล็ก",
        "Screening assistant — ระบบเสนอข้อมูลให้ครบ แต่ให้ผู้ใช้เป็นผู้ตัดสินใจสุดท้าย",
    ])

    doc.save()


if __name__ == "__main__":
    main()
