"""
สร้างเอกสาร UML / SA รวม 10 ไดอาแกรม -> docs/uml_sa_diagrams.docx
+ รูป PNG ใน figures/diagrams/uml_*.png

    python make_uml_doc.py

ต้องมี: python-docx, matplotlib

หมายเหตุ: ระบบนี้เป็น CLI + prototype UI (Gradio) + โมเดล ML — ไม่มี DB / login / บัญชีผู้ใช้
ERD = schema ของไฟล์ JSON/CSV, Class = โมดูล+โครงสร้าง dict (โค้ด functional),
State = วงจรชีวิตของภาพระหว่างประมวลผล
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Ellipse, Circle,
                                Rectangle, Polygon)
from matplotlib import font_manager

BASE = Path(__file__).resolve().parent
DIA = BASE / "figures" / "diagrams"
DOCX = BASE / "docs" / "uml_sa_diagrams.docx"

for _n in ("Tahoma", "Leelawadee UI", "Angsana New"):
    try:
        font_manager.findfont(_n, fallback_to_default=False)
        plt.rcParams["font.family"] = _n
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

CL = {"a": "#E8EAF0", "s1": "#CFE2F3", "s2": "#D9EAD3", "data": "#FCE5CD",
      "out": "#EAD1DC", "note": "#FFF2CC", "ext": "#F0F0F0", "sys": "#DAE8FC"}


def new(Y, w=11):
    """แคนวาส: x 0..100, y 0..Y  (title อยู่แถว Y-2, เนื้อหาอยู่ใต้ Y-8)"""
    fig, ax = plt.subplots(figsize=(w, max(3.5, w * Y / 100 * 0.92)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, Y)
    ax.axis("off")
    return fig, ax


def title(ax, t, sub=None):
    Y = ax.get_ylim()[1]
    ax.text(50, Y - 1.5, t, ha="center", va="top", fontsize=12, fontweight="bold")
    if sub:
        ax.text(50, Y - 6.0, sub, ha="center", va="top", fontsize=7.6, color="#666")


def box(ax, x, y, w, h, text, fc="#FFFFFF", fs=8.5, round_=True, ec="#555"):
    style = "round,pad=0.6,rounding_size=1.6" if round_ else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=5)
    # cx,by, cx,ty, lx,lym, rx,rym
    return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)


def oval(ax, cx, cy, w, h, text, fc=CL["s2"], fs=8):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=fc, ec="#555", lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=5)
    return (cx, cy - h / 2, cx, cy + h / 2, cx - w / 2, cy, cx + w / 2, cy)


def actor(ax, x, y, name, s=5.5):
    ax.add_patch(Circle((x, y + s * 1.7), s * 0.42, fc="white", ec="#333", lw=1.3, zorder=4))
    ax.plot([x, x], [y + s * 1.28, y + s * 0.35], "k-", lw=1.3)
    ax.plot([x - s * 0.7, x + s * 0.7], [y + s * 0.95, y + s * 0.95], "k-", lw=1.3)
    ax.plot([x, x - s * 0.55], [y + s * 0.35, y - s * 0.45], "k-", lw=1.3)
    ax.plot([x, x + s * 0.55], [y + s * 0.35, y - s * 0.45], "k-", lw=1.3)
    ax.text(x, y - s * 1.0, name, ha="center", va="center", fontsize=7.6, fontweight="bold")
    return (x, y + s * 0.7)          # จุดต่อสายกลางลำตัว


def diamond(ax, cx, cy, w, h, text, fc=CL["note"], fs=7.2):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
                          (cx - w / 2, cy)], closed=True, fc=fc, ec="#555", lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=5)
    return (cx, cy - h / 2, cx, cy + h / 2, cx - w / 2, cy, cx + w / 2, cy)


def arrow(ax, p1, p2, dashed=False, txt=None, fs=7.0, tx_dy=1.8, color="#333", head="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=head, mutation_scale=12, lw=1.1,
                                 color=color, ls="--" if dashed else "-",
                                 shrinkA=0, shrinkB=2))
    if txt:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + tx_dy
        ax.text(mx, my, txt, fontsize=fs, ha="center", va="center", color="#333",
                bbox=dict(fc="white", ec="none", pad=0.5), zorder=6)


def save(fig, name):
    DIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIA / name, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print("  ", (DIA / name).relative_to(BASE))
    return name


# ============ 1. Use Case ============
def d1():
    fig, ax = new(80, 13)
    title(ax, "1. Use Case Diagram", "ระบบตรวจจับตำหนิพื้นผิวเหล็ก (2-Stage)")
    ax.add_patch(Rectangle((21, 4), 62, 66, fc="none", ec="#888", lw=1.2))
    ax.text(52, 67, "ระบบตรวจจับตำหนิพื้นผิวเหล็ก", ha="center", fontsize=8, style="italic")
    op = actor(ax, 8, 36, "ผู้ใช้งาน\n(Operator)")
    eng = actor(ax, 95, 36, "ผู้พัฒนา\n(ML Engineer)")
    L = [oval(ax, 40, y, 24, 7, t) for y, t in [
        (60, "อัปโหลดภาพเหล็ก"), (50, "ปรับ confidence /\nโหมดตรวจละเอียด"),
        (40, "ตรวจภาพ (Stage 1 + Stage 2)"), (30, "ดูผล: กรอบ + ชนิด + ความเสี่ยง"),
        (20, "บันทึก / ส่งออกผล (JSON, ภาพ)")]]
    R = [oval(ax, 66, y, 24, 7, t, CL["data"]) for y, t in [
        (63, "เตรียม dataset"), (53, "เทรนโมเดล Stage 2"), (43, "วัดผล (mAP / F1)"),
        (33, "ปรับ per-class threshold"), (23, "วัด Stage 1 / วัดกับภาพจริง"),
        (13, "สร้างเอกสาร / ไดอาแกรม")]]
    for u in L:
        arrow(ax, (op[0] + 2, op[1]), (u[4], u[5]), head="-")
    for u in R:
        arrow(ax, (eng[0] - 2, eng[1]), (u[6], u[7]), head="-")
    arrow(ax, (L[2][2], L[2][3]), (L[0][0], L[0][1]), dashed=True, txt="«include»", fs=6.0)
    arrow(ax, (L[2][0], L[2][1]), (L[3][2], L[3][3]), dashed=True, txt="«include»", fs=6.0)
    return save(fig, "uml_1_usecase.png")


# ============ 2. Context ============
def d2():
    fig, ax = new(82, 12)
    title(ax, "2. Context Diagram (DFD Level 0)")
    c = oval(ax, 50, 40, 26, 15, "ระบบตรวจจับ\nตำหนิพื้นผิวเหล็ก", CL["sys"], 9)
    E = [
        (4, 58, "ผู้ใช้งาน", "ภาพ + พารามิเตอร์  →\n←  ภาพผล + รายการตำหนิ", "r"),
        (4, 14, "ผู้พัฒนา", "dataset + คำสั่ง  →\n←  metric, thresholds.json", "r"),
        (72, 60, "Apple DMS46", "DMS46_v1.pt  →", "l"),
        (72, 40, "Ultralytics YOLO", "yolo11s.pt + ไลบรารี  →", "l"),
        (72, 20, "Roboflow (เตรียมข้อมูล)", "NEU / Rust / Crack  →", "l"),
        (37, 4, "Local File System", "↔  weights / results", "b"),
    ]
    for x, y, nm, lbl, side in E:
        b = box(ax, x, y, 24, 12, nm, CL["ext"], 7.4, round_=False)
        if side == "r":
            arrow(ax, (b[6], b[7]), (c[2] if y > 40 else c[2], c[1] if y > 40 else c[0]), txt=lbl, fs=6.0)
        elif side == "l":
            arrow(ax, (b[4], b[5]), (c[3], c[1]), txt=lbl, fs=6.0)
        else:
            arrow(ax, (b[0], b[3]), (c[0], c[0]), txt=lbl, fs=6.0)
    return save(fig, "uml_2_context.png")


# ============ 3. C4 L1 ============
def d3():
    fig, ax = new(74, 12)
    title(ax, "3. C4 Model — Level 1: System Context")
    op = box(ax, 6, 48, 21, 12, "ผู้ใช้งาน [Person]\nพนักงานคัดกรองเหล็ก", CL["a"], 7.3)
    eng = box(ax, 6, 12, 21, 12, "ผู้พัฒนา [Person]\nดูแลโมเดล/ข้อมูล", CL["a"], 7.3)
    sysb = box(ax, 39, 30, 24, 16,
               "Steel Surface Defect\nDetection System\n[Software System]\n2-stage: DMS46 → YOLO11s", CL["sys"], 7.6)
    dms = box(ax, 76, 50, 20, 11, "Apple DMS46\n[External]\nmaterial segmentation", CL["ext"], 7.1)
    yolo = box(ax, 76, 31, 20, 11, "Ultralytics YOLO\n[External]\ndetector runtime", CL["ext"], 7.1)
    fs_ = box(ax, 76, 12, 20, 11, "Local File System\n[External]\nweights / results", CL["ext"], 7.1)
    arrow(ax, (op[6], op[7]), (sysb[4], sysb[5] + 3), txt="อัปโหลดภาพ / ดูผล\n[HTTP, Gradio]", fs=6.0)
    arrow(ax, (eng[6], eng[7]), (sysb[4], sysb[5] - 3), txt="เทรน / วัดผล [CLI]", fs=6.0)
    arrow(ax, (sysb[6], sysb[7] + 3), (dms[4], dms[5]), txt="inference [TorchScript]", fs=6.0)
    arrow(ax, (sysb[6], sysb[7]), (yolo[4], yolo[5]), txt="inference [Python]", fs=6.0)
    arrow(ax, (sysb[6], sysb[7] - 3), (fs_[4], fs_[5]), txt="อ่าน/เขียนไฟล์", fs=6.0)
    return save(fig, "uml_3_c4_l1.png")


# ============ 4. C4 L2 ============
def d4():
    fig, ax = new(92, 12.5)
    title(ax, "4. C4 Model — Level 2: Container")
    ax.add_patch(Rectangle((22, 4), 74, 74, fc="none", ec="#8ca", lw=1.4, ls="--"))
    ax.text(59, 80, "Steel Surface Defect Detection System", ha="center", fontsize=7.6, style="italic")
    op = box(ax, 2, 52, 16, 12, "ผู้ใช้งาน\n[Person]", CL["a"], 7.2)
    eng = box(ax, 2, 16, 16, 12, "ผู้พัฒนา\n[Person]", CL["a"], 7.2)
    ui = box(ax, 26, 56, 24, 12, "Web UI\n[Gradio/Browser]\napp.py", CL["s2"], 7.0)
    cli = box(ax, 26, 38, 24, 12, "CLI\n[Python]\npipeline.py __main__", CL["s2"], 7.0)
    core = box(ax, 56, 46, 24, 13, "Pipeline Core\n[pipeline.py]\norchestrate + NMS + threshold", CL["sys"], 7.0)
    st1 = box(ax, 56, 63, 24, 11, "Stage 1 Runtime\n[PyTorch/TorchScript]\nDMS46", CL["s1"], 7.0)
    st2 = box(ax, 56, 30, 24, 11, "Stage 2 Runtime\n[Ultralytics YOLO]\nYOLO11s", CL["s1"], 7.0)
    store = box(ax, 56, 9, 24, 12, "Model & Config Store\n[Local FS]\nDMS46_v1.pt · best.pt ·\nthresholds.json", CL["data"], 6.6)
    scr = box(ax, 26, 9, 24, 12, "Training/Eval Scripts\n[Python]\ntrain.py · evaluate*.py ·\ntune_thresholds.py", CL["data"], 6.6)
    arrow(ax, (op[6], op[7]), (ui[4], ui[5]), txt="[HTTP]", fs=6.0)
    arrow(ax, (eng[6], eng[7]), (cli[4], cli[5]), txt="[shell]", fs=6.0)
    arrow(ax, (eng[6], eng[7] - 3), (scr[4], scr[5]), fs=6.0)
    arrow(ax, (ui[6], ui[7]), (core[4], core[3] - 2), txt="analyze()", fs=6.0)
    arrow(ax, (cli[6], cli[7]), (core[4], core[5]), txt="process_image()", fs=6.0)
    arrow(ax, (core[0], core[3]), (st1[0], st1[1]), txt="run_stage1", fs=6.0)
    arrow(ax, (core[0], core[1]), (st2[0], st2[3]), txt="run_stage2", fs=6.0)
    arrow(ax, (core[0], core[1]), (store[0], store[3]), dashed=True, txt="โหลด weights / thresholds", fs=5.8)
    arrow(ax, (scr[6], scr[5]), (store[4], store[5]), dashed=True, txt="เขียน best.pt / thresholds.json", fs=5.8)
    return save(fig, "uml_4_c4_l2.png")


# ============ 5. ERD ============
def d5():
    fig, ax = new(96, 12.5)
    title(ax, "5. ER Diagram — schema ของผลลัพธ์ + ข้อมูลเทรน",
          "ไม่มีฐานข้อมูล — เอนทิตี = โครงสร้างไฟล์ *_result.json / label(.txt) / labels.csv")

    def ent(x, ytop, name, attrs, fc):
        box(ax, x, ytop - 6, 27, 6, name, fc, 8.0, round_=False)
        for i, a in enumerate(attrs):
            ax.add_patch(Rectangle((x, ytop - 6 - (i + 1) * 4.4), 27, 4.4, fc="white", ec="#999", lw=0.6))
            ax.text(x + 1.5, ytop - 6 - i * 4.4 - 2.2, "• " + a, fontsize=6.4, va="center")
        bot = ytop - 6 - len(attrs) * 4.4
        return {"cx": x + 13.5, "top": ytop, "bot": bot, "l": x, "r": x + 27, "mid": (ytop + bot) / 2}

    R = ent(3, 84, "RESULT", ["image (path)", "metal_regions", "metal_area_ratio", "fallback_full_image"], CL["out"])
    RG = ent(3, 52, "REGION", ["region_id (PK)", "box_xywh"], CL["s2"])
    D = ent(38, 74, "DETECTION", ["class (FK)", "confidence", "bbox_xywh",
                                  "bbox_xyxy_crop", "bbox_xyxy_global", "region_id (FK)"], CL["s1"])
    DC = ent(72, 66, "DEFECT_CLASS", ["id 0–7 (PK)", "name", "name_th", "source", "risk"], CL["note"])
    DI = ent(38, 28, "DATASET_IMAGE", ["filename (PK)", "split", "width", "height"], CL["data"])
    LB = ent(72, 26, "LABEL_BOX", ["class_id (FK)", "xc, yc, w, h"], CL["data"])
    RT = ent(3, 24, "REAL_TEST_LABEL", ["filename (PK)", "classes[] (image-level)"], CL["ext"])
    arrow(ax, (R["cx"], R["bot"]), (RG["cx"], RG["top"]), head="-", txt="1 : N", fs=6.3)
    arrow(ax, (RG["r"], RG["mid"]), (D["l"], D["mid"]), head="-", txt="1 : N", fs=6.3)
    arrow(ax, (D["r"], D["mid"]), (DC["l"], DC["mid"]), head="-", txt="N : 1", fs=6.3)
    arrow(ax, (DI["r"], DI["mid"]), (LB["l"], LB["mid"]), head="-", txt="1 : N", fs=6.3)
    arrow(ax, (LB["cx"], LB["top"]), (DC["cx"], DC["bot"]), head="-", txt="N : 1", fs=6.3)
    return save(fig, "uml_5_erd.png")


# ============ 6. Class ============
def d6():
    fig, ax = new(120, 12)
    title(ax, "6. Class Diagram — โมดูล + โครงสร้างข้อมูล",
          "โค้ด functional — «module» = ไฟล์ .py, «data» = dict ที่ส่งต่อกันใน pipeline")
    LINE, HEAD = 3.4, 6.0

    def cls(x, ytop, w, name, members, fc):
        h = HEAD + LINE * len(members) + 1.5
        ax.add_patch(Rectangle((x, ytop - h), w, h, fc=fc, ec="#555", lw=1.1))
        ax.plot([x, x + w], [ytop - HEAD, ytop - HEAD], "k-", lw=0.7)
        ax.text(x + w / 2, ytop - HEAD / 2, name, ha="center", fontsize=7.4, fontweight="bold")
        for i, m in enumerate(members):
            ax.text(x + 1.4, ytop - HEAD - 2.4 - i * LINE, m, fontsize=6.0, va="center")
        return {"cx": x + w / 2, "top": ytop, "bot": ytop - h, "l": x, "r": x + w}

    P = cls(3, 114, 42, "«module» pipeline", [
        "+ resolve_device(req)", "+ load_models(dev) : (s1, s2)",
        "+ run_stage1(m, img, dev) : mask", "+ _preprocess_for_dms(img)",
        "+ mask_to_boxes(mask) : boxes", "+ _merge_close_boxes(boxes, gap)",
        "+ build_regions(mask, shape) : (boxes, meta)",
        "+ run_stage2(m, crop, conf, aug, class_conf)",
        "+ cross_region_nms(dets, iou)", "+ load_class_conf(path)",
        "+ process_image(...) : summary", "+ draw_thai_text(img, txt, pos)"], CL["sys"])
    Ap = cls(3, P["bot"] - 6, 42, "«module» app  (Gradio UI)", [
        "- _STATE : {s1, s2, device, class_conf}", "+ analyze(img, conf, detailed)",
        "+ build_ui() : Blocks"], CL["a"])
    Ev = cls(3, Ap["bot"] - 6, 42, "«module» evaluate / evaluate_real /\nevaluate_stage1 / tune_thresholds",
             ["ใช้ pipeline.* ทั้งหมด"], CL["data"])

    De = cls(52, 114, 45, "«data» Detection", [
        "class : str", "confidence : float", "bbox_xywh : [4]",
        "bbox_xyxy_crop : [4]", "bbox_xyxy_global : [4]", "region_id : int"], CL["s1"])
    Me = cls(52, De["bot"] - 6, 45, "«data» RegionMeta", [
        "metal_found : bool", "metal_ratio : float",
        "fallback_full_image : bool", "n_regions : int"], CL["s2"])
    Su = cls(52, Me["bot"] - 6, 45, "«data» Summary", [
        "image : str", "metal_regions : int", "metal_area_ratio : float",
        "regions : [ {box_xywh, detections[]} ]"], CL["out"])
    Co = cls(52, Su["bot"] - 6, 45, "«const»", [
        "DEFECT_CLASSES : [8]", "DEFECT_INFO : {name_th, risk}",
        "METAL_MODEL_INDEX = 22"], CL["note"])

    arrow(ax, (Ap["r"], Ap["top"] - 3), (P["l"], P["bot"] + 3), dashed=True, txt="uses", fs=6.0)
    arrow(ax, (Ev["r"], Ev["top"] - 3), (P["l"], P["bot"] + 8), dashed=True, txt="uses", fs=6.0)
    arrow(ax, (P["r"], De["bot"] + 10), (De["l"], De["bot"] + 10), dashed=True, txt="creates", fs=6.0)
    arrow(ax, (P["r"], Me["top"] - 3), (Me["l"], Me["top"] - 3), dashed=True, txt="creates", fs=6.0)
    arrow(ax, (P["r"], Su["top"] - 3), (Su["l"], Su["top"] - 3), dashed=True, txt="creates", fs=6.0)
    return save(fig, "uml_6_class.png")


# ============ 7. Sequence ============
def d7():
    fig, ax = new(96, 13)
    title(ax, "7. Sequence Diagram — ตรวจภาพผ่าน UI")
    lanes = ["ผู้ใช้งาน", "Web UI\n(app.py)", "Pipeline\n(pipeline.py)",
             "DMS46\n(Stage 1)", "YOLO11s\n(Stage 2)", "File System"]
    xs = [9, 27, 46, 64, 80, 94]
    hdr, bot = 84, 6
    for x, nm in zip(xs, lanes):
        box(ax, x - 7, hdr, 14, 6, nm, CL["a"], 6.8, round_=False)
        ax.plot([x, x], [hdr, bot], color="#999", lw=1, ls=(0, (4, 3)))

    def m(i, j, y, t, dashed=False):
        arrow(ax, (xs[i], y), (xs[j], y), dashed=dashed, txt=t, fs=6.1, tx_dy=1.5)

    def note(y, t):
        ax.text(46, y, t, fontsize=6.0, ha="center",
                bbox=dict(fc=CL["note"], ec="#999", pad=1.4), zorder=6)

    y = 78
    m(0, 1, y, "อัปโหลดภาพ + ตั้ง conf, กด 'ตรวจสอบ'"); y -= 6
    m(1, 2, y, "analyze(image, conf, detailed)"); y -= 6
    m(2, 5, y, "load_models()  [ครั้งแรก]", True); y -= 5
    m(5, 2, y, "DMS46_v1.pt · best.pt · thresholds.json", True); y -= 7
    m(2, 3, y, "run_stage1(image)"); y -= 5
    m(3, 2, y, "metal mask", True); y -= 7
    note(y, "build_regions()  —  [metal_ratio < 0.05 → เพิ่มกรอบทั้งภาพ]"); y -= 7
    ax.add_patch(Rectangle((43, y - 20), 6, 22, fc="#ececec", ec="#999", lw=0.7))
    ax.text(39, y - 9, "loop\n[แต่ละ region]", fontsize=5.8, ha="center")
    m(2, 4, y, "run_stage2(crop, conf, class_conf)"); y -= 5
    m(4, 2, y, "detections", True); y -= 9
    note(y, "cross_region_nms()  +  กรอง per-class threshold"); y -= 8
    m(2, 1, y, "annotated image + table + verdict", True); y -= 6
    m(1, 0, y, "แสดงผลบนหน้าจอ", True)
    return save(fig, "uml_7_sequence.png")


# ============ 8. Activity ============
def d8():
    fig, ax = new(192, 8.5)
    title(ax, "8. Activity Diagram — การประมวลผล 1 ภาพ")
    ax.add_patch(Circle((50, 184), 2.2, fc="black"))
    y = 182                                   # ขอบบนของกล่องถัดไป
    prev = (50, 184)
    for fc, t in [("a", "อ่านภาพ (cv2.imread)"),
                  ("s1", "Stage 1: DMS46  →  สร้าง metal mask"),
                  ("s1", "mask_to_boxes(): contour + รวมกรอบ + กรอง")]:
        b = box(ax, 22, y - 7, 56, 7, t, CL[fc], 7.2)
        arrow(ax, prev, (b[0], b[3]))
        prev = (b[0], b[1]); y -= 12
    d = diamond(ax, 50, y - 7, 44, 14, "metal_ratio < 0.05\nหรือไม่เจอกรอบ?")
    arrow(ax, prev, (d[2], d[3]))
    bf = box(ax, 4, y - 11, 30, 7, "เพิ่มกรอบ = ทั้งภาพ\n(fallback)", CL["note"], 6.8)
    arrow(ax, (d[4], d[5]), (bf[6], bf[7]), txt="ใช่", fs=6.2)
    mrg = y - 16
    arrow(ax, (d[0], d[1]), (50, mrg + 1.4), txt="ไม่ใช่", fs=6.2)
    arrow(ax, (bf[0], bf[1]), (50, mrg + 1.4))
    ax.add_patch(Circle((50, mrg), 1.5, fc="black"))
    y = mrg - 4
    for fc, t in [("s2", "loop [แต่ละ region]: crop  →  [ถ้าโมเดล gray → แปลงเป็น gray]"),
                  ("s2", "YOLO11s.predict  →  map bbox เป็นพิกัดภาพเต็ม"),
                  ("s2", "รวมทุก region  →  cross_region_nms()"),
                  ("s2", "กรอง per-class threshold (thresholds.json)")]:
        b = box(ax, 14, y - 7, 72, 7, t, CL[fc], 7.0)
        arrow(ax, (50, y + 2.5), (b[0], b[3]))
        y -= 12
    d2 = diamond(ax, 50, y - 6, 34, 12, "มี detection?")
    arrow(ax, (50, y + 2.5), (d2[2], d2[3]))
    by = box(ax, 6, y - 20, 40, 7, "วาดกรอบ + ป้ายไทย + ความเสี่ยง", CL["s2"], 6.9)
    bn = box(ax, 54, y - 20, 40, 7, "ทำเครื่องหมาย \"ปกติ\"", CL["a"], 6.9)
    arrow(ax, (d2[4], d2[5]), (by[6], by[7]), txt="ใช่", fs=6.2)
    arrow(ax, (d2[6], d2[7]), (bn[4], bn[5]), txt="ไม่", fs=6.2)
    jn = y - 27
    arrow(ax, (by[0], by[1]), (50, jn + 1.4))
    arrow(ax, (bn[0], bn[1]), (50, jn + 1.4))
    ax.add_patch(Circle((50, jn), 1.5, fc="black"))
    bs = box(ax, 22, jn - 11, 56, 7, "บันทึก *_result.jpg + *_result.json", CL["out"], 7.0)
    arrow(ax, (50, jn - 1.5), (bs[0], bs[3]))
    arrow(ax, (bs[0], bs[1]), (50, jn - 17))
    ax.add_patch(Circle((50, jn - 19), 2.4, fc="none", ec="black", lw=1.4))
    ax.add_patch(Circle((50, jn - 19), 1.1, fc="black"))
    return save(fig, "uml_8_activity.png")


# ============ 9. State ============
def d9():
    fig, ax = new(52, 12)
    title(ax, "9. State Diagram — สถานะของภาพระหว่างประมวลผล",
          "ระบบไม่มี session/entity ที่คงสถานะข้ามคำขอ — แสดงวงจรชีวิตของภาพ 1 รูปในหน่วยความจำ")
    ax.add_patch(Circle((5, 30), 1.7, fc="black"))
    xs = [10, 29, 48, 67, 84]
    labs = ["Loaded\nอ่านภาพแล้ว", "Stage1Done\nได้ metal mask",
            "RegionsReady\n(+ fallback ถ้าต้อง)", "Stage2Done\nraw detections",
            "Filtered\nผ่าน NMS + threshold"]
    prev = (6.7, 30)
    b = None
    for x, t in zip(xs, labs):
        b = box(ax, x, 24, 15, 11, t, CL["s1"], 6.8)
        arrow(ax, prev, (b[4], b[5])); prev = (b[6], b[5])
    fin = box(ax, 39, 6, 22, 9, "Rendered & Saved\n*_result.jpg / .json", CL["out"], 6.8)
    arrow(ax, (b[0], b[1]), (fin[6], fin[7]))
    arrow(ax, (fin[4], fin[5]), (55, 24), dashed=True, txt="[ภาพถัดไป]", fs=6.0)
    ax.add_patch(Circle((30, 8), 2.1, fc="none", ec="black", lw=1.3))
    ax.add_patch(Circle((30, 8), 1.0, fc="black"))
    arrow(ax, (fin[4], fin[5] - 1), (32, 8))
    return save(fig, "uml_9_state.png")


# ============ 10. Deployment ============
def d10():
    fig, ax = new(88, 12)
    title(ax, "10. Deployment Diagram", "รันบนเครื่องเดียว (ไม่มี server / cloud)")
    ax.add_patch(Rectangle((5, 5), 90, 72, fc="#F7F7F7", ec="#555", lw=1.4))
    ax.text(50, 73, "«device»  Windows 11 Laptop", ha="center", fontsize=8.5, fontweight="bold")
    box(ax, 63, 54, 28, 12, "«device»\nNVIDIA RTX 3050 6GB\nCUDA 12.4", CL["ext"], 7.0)
    ax.add_patch(Rectangle((9, 22), 46, 44, fc="white", ec="#777", lw=1.1))
    ax.text(32, 62, "«execution environment»  Python 3.11 venv", ha="center", fontsize=7.0, fontweight="bold")
    for i, (t, c) in enumerate([
        ("«artifact» pipeline.py / app.py", CL["s2"]),
        ("«artifact» PyTorch 2.6 + Ultralytics 8.4", CL["s1"]),
        ("«artifact» DMS46_v1.pt", CL["data"]),
        ("«artifact» runs/detect/train-gray-s/best.pt", CL["data"]),
        ("«artifact» thresholds.json", CL["note"])]):
        box(ax, 11, 54 - i * 7, 42, 5.4, t, c, 6.3, round_=False)
    ax.add_patch(Rectangle((60, 26), 30, 20, fc="white", ec="#777", lw=1.1))
    ax.text(75, 42, "«execution environment»  Web Browser", ha="center", fontsize=6.8, fontweight="bold")
    box(ax, 62, 29, 26, 8, "Gradio client\nhttp://127.0.0.1:7860", CL["a"], 6.5, round_=False)
    box(ax, 18, 9, 64, 8, "«file system»  merged_dataset_gray/ · real_test/ · pipeline_results/ · results/",
        CL["ext"], 6.4, round_=False)
    arrow(ax, (74, 29), (33, 52), dashed=True, txt="HTTP :7860", fs=6.0)
    arrow(ax, (32, 22), (42, 17), dashed=True, txt="อ่าน/เขียนไฟล์", fs=6.0)
    arrow(ax, (55, 48), (63, 54), dashed=True, txt="CUDA", fs=6.0)
    return save(fig, "uml_10_deployment.png")


def build_docx(items):
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    doc.styles["Normal"].font.name = "Tahoma"
    doc.styles["Normal"].font.size = Pt(11)
    h = doc.add_heading("UML / SA Diagrams — ระบบตรวจจับตำหนิพื้นผิวเหล็ก (2-Stage)", 0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "ระบบเป็น CLI + prototype UI (Gradio) + โมเดล ML — ไม่มีฐานข้อมูล / ระบบล็อกอิน / บัญชีผู้ใช้. "
        "ไดอาแกรมที่ผูกกับ DB/CRUD ถูกปรับให้ตรงกับของจริง: ERD = schema ของไฟล์ JSON/label, "
        "Class = โมดูล + โครงสร้าง dict (โค้ด functional), State = วงจรชีวิตของภาพระหว่างประมวลผล. "
        "สร้างจาก make_uml_doc.py").alignment = WD_ALIGN_PARAGRAPH.CENTER

    names = {
        "uml_1_usecase.png": "1. Use Case Diagram",
        "uml_2_context.png": "2. Context Diagram (DFD Level 0)",
        "uml_3_c4_l1.png": "3. C4 Model — Level 1: System Context",
        "uml_4_c4_l2.png": "4. C4 Model — Level 2: Container",
        "uml_5_erd.png": "5. ER Diagram (schema ของ JSON / label)",
        "uml_6_class.png": "6. Class Diagram (โมดูล + โครงสร้างข้อมูล)",
        "uml_7_sequence.png": "7. Sequence Diagram — ตรวจภาพผ่าน UI",
        "uml_8_activity.png": "8. Activity Diagram — การประมวลผล 1 ภาพ",
        "uml_9_state.png": "9. State Diagram — สถานะของภาพระหว่างประมวลผล",
        "uml_10_deployment.png": "10. Deployment Diagram",
    }
    desc = {
        "uml_1_usecase.png": "Actor 2 ราย — ผู้ใช้งาน (อัปโหลดภาพ, ปรับพารามิเตอร์, ตรวจภาพ, ดู/บันทึกผล) "
        "และผู้พัฒนา (เตรียม dataset, เทรน, วัดผล, ปรับ threshold, สร้างเอกสาร). \"ตรวจภาพ\" «include» การอัปโหลดและการดูผล.",
        "uml_2_context.png": "ระบบรับภาพ+พารามิเตอร์จากผู้ใช้ คืนภาพผล+รายการตำหนิ; รับ dataset/คำสั่งจากผู้พัฒนา "
        "คืน metric/thresholds.json; พึ่งไฟล์โมเดล DMS46, ไลบรารี YOLO, ระบบไฟล์ท้องถิ่น. Roboflow เกี่ยวเฉพาะตอนเตรียมข้อมูล (offline).",
        "uml_3_c4_l1.png": "มุมมอง C4 สูงสุด: ผู้ใช้/ผู้พัฒนา ↔ ระบบ ↔ ระบบภายนอก (DMS46, YOLO runtime, file system). "
        "ไม่มีการเรียก API ภายนอกผ่านเครือข่าย.",
        "uml_4_c4_l2.png": "Container ภายในระบบ: Web UI (Gradio), CLI, Pipeline Core (ตัวประสาน), Stage 1 Runtime "
        "(PyTorch/TorchScript), Stage 2 Runtime (Ultralytics), Model & Config Store (ไฟล์บนดิสก์), ชุดสคริปต์เทรน/วัดผล. ไม่มี container ฐานข้อมูล.",
        "uml_5_erd.png": "ไม่มี DB — เอนทิตีคือโครงสร้างไฟล์: RESULT 1:N REGION 1:N DETECTION N:1 DEFECT_CLASS; "
        "ฝั่งข้อมูลเทรน DATASET_IMAGE 1:N LABEL_BOX N:1 DEFECT_CLASS; REAL_TEST_LABEL = label ระดับภาพ (multi-label) สำหรับ evaluate_real.py.",
        "uml_6_class.png": "โค้ด functional — «module» pipeline รวมฟังก์ชันหลัก, «data» Detection/RegionMeta/Summary "
        "คือ dict ที่ไหลผ่าน pipeline, app และ evaluate* เป็นโมดูลที่ใช้ pipeline.",
        "uml_7_sequence.png": "ลำดับเมื่อผู้ใช้กด 'ตรวจสอบ': analyze() → load_models() (ครั้งแรก) → run_stage1 → "
        "build_regions (อาจ fallback) → loop run_stage2 ต่อ region → cross_region_nms + กรอง threshold → คืนภาพ+ตาราง+สรุปผล.",
        "uml_8_activity.png": "Flow การประมวลผล 1 ภาพ พร้อมจุดตัดสินใจ 2 จุด: (1) metal_ratio < 0.05 → เพิ่มกรอบทั้งภาพ  "
        "(2) มี detection ไหม → วาดผล / ทำเครื่องหมายปกติ แล้วบันทึกไฟล์.",
        "uml_9_state.png": "วงจรชีวิตของภาพในหน่วยความจำ: Loaded → Stage1Done → RegionsReady → Stage2Done → Filtered "
        "→ Rendered & Saved. ระบบไม่มี session ที่คงสถานะข้ามคำขอ.",
        "uml_10_deployment.png": "รันบนโน้ตบุ๊ก Windows 11 เครื่องเดียว: Python venv (โค้ด + PyTorch/Ultralytics + ไฟล์โมเดล + "
        "thresholds.json), เบราว์เซอร์ต่อ Gradio ที่ 127.0.0.1:7860, GPU RTX 3050 ผ่าน CUDA, ข้อมูล/ผลลัพธ์บนระบบไฟล์ท้องถิ่น. ไม่มี server/cloud.",
    }
    for img in items:
        doc.add_page_break()
        doc.add_heading(names[img], level=1)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(DIA / img), width=Inches(6.6))
        doc.add_paragraph(desc[img])
    DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX))
    print("\nเขียน:", DOCX.relative_to(BASE))


if __name__ == "__main__":
    print("สร้าง UML/SA diagrams -> figures/diagrams/")
    items = [d1(), d2(), d3(), d4(), d5(), d6(), d7(), d8(), d9(), d10()]
    build_docx(items)
