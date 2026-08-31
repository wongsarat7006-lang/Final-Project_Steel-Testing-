"""
รวม 3 dataset (NEU, Rust, Crack) เข้าด้วยกันเป็น dataset เดียว 8 classes

Class ใหม่ทั้งหมด:
    0: crazing          (จาก NEU)
    1: inclusion         (จาก NEU)
    2: patches           (จาก NEU)
    3: pitted_surface    (จาก NEU)
    4: rolled-in_scale   (จาก NEU)
    5: scratches         (จาก NEU)
    6: rust              (จาก Rust dataset: รวม RUST + DANGER-RUST, ตัด NO-RUST ทิ้ง)
    7: crack             (จาก Crack dataset)

วิธีใช้:
    python merge_datasets.py
"""
import os
import random
import shutil
import glob

# ===== ตั้งค่า path ของแต่ละ dataset (แก้ตามจริงถ้าจำเป็น) =====
NEU_DIR = "."
RUST_DIR = "rust_dataset"
CRACK_DIR = "crack_dataset"
OUTPUT_DIR = "merged_dataset"

# NEU ใช้ class id 0-5 อยู่แล้ว ไม่ต้องแปลง
# Rust: RUST(2)->6, DANGER-RUST(0)->6, NO-RUST(1)-> ตัดทิ้ง
RUST_CLASS_MAP = {0: 6, 2: 6}  # DANGER-RUST->6, RUST->6, (1=NO-RUST ไม่อยู่ใน map เลยจะถูกตัดทิ้ง)
# Crack: crack(0)->7
CRACK_CLASS_MAP = {0: 7}

NEW_CLASS_NAMES = [
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
    "rust", "crack",
]


def _line_to_bbox(parts):
    """แปลง 1 บรรทัด label ให้เป็น bbox YOLO: 'x_c y_c w h' (normalized)
    - บรรทัด detect (4 ค่า) : คืนค่าเดิม
    - บรรทัด segment/polygon (คู่ x,y ตั้งแต่ 3 จุด) : คืน bounding box ที่ครอบ polygon
    """
    coords = [float(v) for v in parts]
    if len(coords) == 4:
        return coords
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]


def remap_label_file(src_label_path, dst_label_path, class_map):
    """อ่านไฟล์ label เดิม แปลง class id ตาม class_map แล้วเขียนไฟล์ใหม่
    - บรรทัดที่ class id ไม่อยู่ใน class_map จะถูกตัดทิ้ง (เช่น NO-RUST)
    - บรรทัด polygon/segment จะถูกแปลงเป็น bounding box (กัน dataset ผสม detect+segment)
    """
    if not os.path.exists(src_label_path):
        # ไม่มีไฟล์ label แปลว่าภาพนี้ไม่มีวัตถุ (background image) - สร้างไฟล์เปล่า
        open(dst_label_path, "w").close()
        return 0

    kept_lines = 0
    with open(src_label_path, "r") as f_in, open(dst_label_path, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            if old_cls not in class_map:
                continue  # ตัดทิ้ง (เช่น NO-RUST)
            new_cls = class_map[old_cls]
            bbox = _line_to_bbox(parts[1:])
            f_out.write(f"{new_cls} " + " ".join(f"{v:.6f}" for v in bbox) + "\n")
            kept_lines += 1
    return kept_lines


# โฟลเดอร์ NEU (train/valid/test ที่ root) ปนภาพ Rust ที่ label ผิดเป็น class 0/1 อยู่ ~932 ไฟล์
# จึงรับเฉพาะไฟล์ที่ชื่อขึ้นต้นด้วยชื่อคลาส NEU จริง (crazing_, inclusion_, ...) เท่านั้น
NEU_VALID_PREFIXES = tuple(
    n + "_" for n in
    ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
)


def copy_neu_split(split):
    """copy ภาพ+label ของ NEU ตรงๆ (class id 0-5 ตรงกับใหม่อยู่แล้ว) พร้อมเติม prefix กันชื่อซ้ำ
    กรองไฟล์ปนเปื้อน (ภาพ Rust ที่หลุดมาอยู่ในโฟลเดอร์ NEU) ออกด้วยการเช็คชื่อไฟล์
    """
    src_img_dir = os.path.join(NEU_DIR, split, "images")
    src_lbl_dir = os.path.join(NEU_DIR, split, "labels")
    dst_img_dir = os.path.join(OUTPUT_DIR, split, "images")
    dst_lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")

    count = 0
    skipped = 0
    for img_path in glob.glob(os.path.join(src_img_dir, "*")):
        filename = os.path.basename(img_path)
        if not filename.lower().startswith(NEU_VALID_PREFIXES):
            skipped += 1
            continue
        new_filename = f"neu_{filename}"
        shutil.copy(img_path, os.path.join(dst_img_dir, new_filename))

        label_name = os.path.splitext(filename)[0] + ".txt"
        src_label = os.path.join(src_lbl_dir, label_name)
        dst_label = os.path.join(dst_lbl_dir, os.path.splitext(new_filename)[0] + ".txt")
        if os.path.exists(src_label):
            # กันไว้อีกชั้น: เก็บเฉพาะบรรทัดที่เป็น class 0-5
            with open(src_label) as f_in, open(dst_label, "w") as f_out:
                for line in f_in:
                    p = line.split()
                    if p and 0 <= int(p[0]) <= 5:
                        f_out.write(line if line.endswith("\n") else line + "\n")
        else:
            open(dst_label, "w").close()
        count += 1
    if skipped:
        print(f"  (ข้ามไฟล์ปนเปื้อนใน NEU/{split}: {skipped} ไฟล์)")
    return count


def copy_and_remap_split(dataset_dir, split, prefix, class_map, limit=None):
    """copy ภาพ + แปลง class id ของ label ให้ตรงกับ dataset ใหม่
    limit: สุ่มเก็บภาพไม่เกิน N ภาพ (ใช้กับ crack ที่มีเยอะเกินสมดุล)
    """
    src_img_dir = os.path.join(dataset_dir, split, "images")
    src_lbl_dir = os.path.join(dataset_dir, split, "labels")
    dst_img_dir = os.path.join(OUTPUT_DIR, split, "images")
    dst_lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")

    img_paths = sorted(glob.glob(os.path.join(src_img_dir, "*")))
    if limit is not None and len(img_paths) > limit:
        rng = random.Random(0)
        img_paths = rng.sample(img_paths, limit)

    count = 0
    skipped_empty = 0
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        new_filename = f"{prefix}_{filename}"
        shutil.copy(img_path, os.path.join(dst_img_dir, new_filename))

        label_name = os.path.splitext(filename)[0] + ".txt"
        src_label = os.path.join(src_lbl_dir, label_name)
        dst_label = os.path.join(dst_lbl_dir, os.path.splitext(new_filename)[0] + ".txt")

        kept = remap_label_file(src_label, dst_label, class_map)
        if kept == 0:
            skipped_empty += 1
        count += 1
    return count, skipped_empty


def main():
    print("===== เริ่มรวม 3 Dataset เข้าด้วยกัน =====\n")

    # ล้าง output เดิมก่อน กันไฟล์ค้างจากการรันครั้งก่อน
    if os.path.isdir(OUTPUT_DIR):
        for split in ["train", "valid", "test"]:
            for sub in ["images", "labels"]:
                d = os.path.join(OUTPUT_DIR, split, sub)
                if os.path.isdir(d):
                    shutil.rmtree(d)
            cache = os.path.join(OUTPUT_DIR, split, "labels.cache")
            if os.path.exists(cache):
                os.remove(cache)

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    # crack มีเยอะกว่าคลาสอื่นมาก จำกัดจำนวนต่อ split ให้พอ ๆ กับคลาส NEU ที่เยอะสุด
    CRACK_LIMIT = {"train": 800, "valid": 150, "test": 150}

    for split in ["train", "valid", "test"]:
        print(f"--- Split: {split} ---")

        neu_count = copy_neu_split(split)
        print(f"  NEU: copy {neu_count} ภาพ")

        rust_count, rust_empty = copy_and_remap_split(RUST_DIR, split, "rust", RUST_CLASS_MAP)
        print(f"  Rust: copy {rust_count} ภาพ (ในจำนวนนี้ {rust_empty} ภาพไม่มีตำหนิเหลืออยู่หลังตัด NO-RUST)")

        crack_count, crack_empty = copy_and_remap_split(
            CRACK_DIR, split, "crack", CRACK_CLASS_MAP, limit=CRACK_LIMIT[split]
        )
        print(f"  Crack: copy {crack_count} ภาพ (จำกัดไว้ที่ {CRACK_LIMIT[split]})")
        print()

    # ใช้ path: เป็น absolute root กัน ultralytics หาโฟลเดอร์ไม่เจอเวลา cwd เปลี่ยน
    abs_root = os.path.abspath(OUTPUT_DIR).replace("\\", "/")
    yaml_content = f"""path: {abs_root}
train: train/images
val: valid/images
test: test/images
nc: {len(NEW_CLASS_NAMES)}
names: {NEW_CLASS_NAMES}
"""
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"===== เสร็จสมบูรณ์ =====")
    print(f"Dataset รวมอยู่ที่: {OUTPUT_DIR}/")
    print(f"data.yaml (8 classes) อยู่ที่: {yaml_path}")


if __name__ == "__main__":
    main()