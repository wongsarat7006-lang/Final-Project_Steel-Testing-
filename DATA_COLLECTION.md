# การเก็บข้อมูลจริง — ทำให้ระบบตรวจภาพลูกค้าได้

> เป้าหมาย: ลูกค้าถ่ายรูปชิ้นเหล็ก → ระบบตีกรอบเฉพาะจุดตำหนิ + บอกชนิด (หลายชนิด/รูปได้)
>
> สถาปัตยกรรม (YOLO + pipeline + UI) **พร้อมแล้ว** — สิ่งเดียวที่ขาดคือ **ข้อมูลเทรนที่หน้าตา
> เหมือนภาพลูกค้า** ชุด NEU-DET/Roboflow ที่ใช้อยู่เป็นภาพแล็บ/สไตล์เฉพาะ โมเดลจึงไม่ยิงบนภาพถ่ายจริง

## ทำไมโหมดความไว / retrain RGB อย่างเดียวไม่พอ

- **โหมดความไว** ลด threshold ได้ แต่ถ้าโมเดลให้ output = 0 (ภาพนอกโดเมนสุด ๆ) ก็ไม่มีอะไรให้ปรับ
- **`--recipe camera` (RGB + aug แรง)** ช่วยเรื่องสี/แสง/มุม แต่ยังเทรนจากภาพชุดเดิม —
  ไม่มีตัวอย่าง "เหล็กเส้นถ่ายบนพื้น" ให้เรียน
- → ต้องมีภาพถ่ายจริงที่ label แล้ว เข้าไปในชุดเทรน

## วิธีที่เร็วที่สุด: Active learning loop

```
ถ่ายภาพ 50-200 รูป  ─►  annotate_bootstrap.py (โมเดลร่าง label)  ─►  คนแก้กรอบ
        ▲                                                                │
        └──────────  retrain (--recipe camera)  ◄─────  merge เข้า train/ ─┘
```
รอบแรกโมเดลร่างมั่ว ๆ คนแก้เยอะ; รอบ 3-4 โมเดลร่างแม่นขึ้น คนแค่ยืนยัน — เร็วขึ้นเรื่อย ๆ

### 1. ถ่ายภาพ (สำคัญสุด — 80% ของงาน)
- **สภาพเหมือนตอนใช้จริง**: มือถือ/กล้องที่จะใช้, แสงจริง, ระยะ ~0.3–1 m, มุมหลากหลาย
- คละ: มีตำหนิ (เน้น rust, crack, scratches, pitted ก่อน) + ปกติ (ไว้กัน false positive)
- 1 ชิ้นมีหลายตำหนิได้ — ถ่ายให้เห็นทุกจุด
- เป้า: **500–1,000 กรอบต่อคลาส** ที่จะรองรับ (สะสมข้ามหลายรอบได้)
- ตั้งชื่อไฟล์ ASCII ไม่มีเว้นวรรค เช่น `real_0001.jpg`

### 2. ร่าง label ด้วยโมเดลปัจจุบัน
```powershell
python annotate_bootstrap.py --src photos_round1/ --out dataset_real/round1 --no-stage1 --conf 0.12
```
ได้ `dataset_real/round1/{images,labels,classes.txt,data.yaml}`

### 3. แก้ label (คน)
เปิด `dataset_real/round1/` ใน **LabelImg** (`pip install labelImg`) หรือ **Label Studio** หรือ import เข้า Roboflow
- ลบกรอบเกิน · ขยับกรอบเพี้ยน · เพิ่มกรอบที่พลาด · แก้ชนิด
- format เป็น YOLO อยู่แล้ว (`classes.txt` มีให้)

### 4. รวมเข้าชุดเทรน + retrain
```powershell
# คัดลอกเข้า train split (หรือแบ่ง val/test ส่วนหนึ่งด้วย resplit_grouped.py)
copy dataset_real\round1\images\* merged_dataset\train\images\
copy dataset_real\round1\labels\* merged_dataset\train\labels\

python make_oversampled_list.py --dataset merged_dataset
python train.py --recipe camera --data merged_dataset/data_oversampled.yaml `
                --model yolo11n.pt --name train-real1 --epochs 120 --batch 8

python evaluate.py --mode stage2 --weights runs/detect/train-real1/weights/best.pt `
                   --data merged_dataset/data.yaml --out results/stage2_real1.json
```

### 5. วนรอบ 2, 3, ...
`annotate_bootstrap.py --weights runs/detect/train-real1/weights/best.pt` บนภาพชุดถัดไป
→ label ร่างแม่นขึ้น → แก้น้อยลง → เทรนใหม่

## เป้าหมายเชิงตัวเลข (ประเมิน)
| รอบ | ภาพสะสม | ผลที่คาด |
|---|---|---|
| 1 | ~150 | เริ่มยิงบนภาพจริงบ้าง (recall ต่ำ) |
| 3 | ~500 | ใช้ได้ระดับคัดกรอง บนสภาพที่เก็บมา |
| 5+ | ~1,500 | เสถียร ครอบคลุมมุม/แสงหลากหลาย |

## หมายเหตุ
- ภาพลูกค้าเล็งชิ้นเดียว → **ไม่ต้องใช้ Stage 1** (`--no-stage1`) เหลือ YOLO ล้วน เร็ว+เสถียรกว่า
- track นี้แยกจาก track เล่มจบ (grayscale + NEU benchmark) — เล่มจบใช้ผลเดิมได้ นี่คือ future work / product
