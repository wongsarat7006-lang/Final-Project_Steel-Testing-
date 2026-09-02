# results/ — ผลการทดลองที่เก็บไว้ (JSON)

ไฟล์ที่ commit = ผลของโมเดล/การทดลองที่รายงานในเล่ม
ไฟล์ `*_latest.json` = output ล่าสุดของสคริปต์ (gitignored — เขียนทับได้เรื่อย ๆ)

| ไฟล์ | มาจาก | คือ |
|---|---|---|
| `stage2_train-clean.json` | `evaluate.py --mode stage2` (train-clean) | mAP50/mAP50-95 รายคลาส บน test |
| `stage2_train-balanced.json` | `evaluate.py --mode stage2` (train-balanced) | เหมือนกัน — โมเดล oversample + texture recipe |
| `stage2_train-gray-s.json` | `evaluate.py` (train-gray-s) | **ยังไม่มี** — สร้างหลังเทรน yolo11s บน dataset ใหม่ |
| `stage1_dms46_test.json` | `evaluate_stage1.py` บน merged_dataset/test | metric ของ Stage 1 (fallback rate / coverage / เวลา) |
| `*_latest.json` | default `--out` ของ evaluate*.py | scratch — คัดลอกไปตั้งชื่อถาวรเองถ้าจะเก็บ |

สร้าง/อัปเดต:

```bash
python evaluate.py --mode stage2 --weights runs/detect/<run>/weights/best.pt \
                   --data <ds>/data.yaml --out results/stage2_<run>.json
python evaluate_stage1.py --dir merged_dataset/test --out results/stage1_dms46_test.json
python make_figures.py    # อ่าน results/stage2_train-clean.json + stage2_train-balanced.json
```
