# external_test/gc10 — GC10-DET (cross-dataset generalization test)

**ที่มา:** GC10-DET (Roboflow export) — https://huggingface.co/datasets/imaadd05/gc10-det
เดิมจาก Lv et al., "Deep Metallic Surface Defect Detection: The New Benchmark and
Detection Network" (GC10-DET), ภาพผิวเหล็กแผ่นจากสายการผลิตจริง 10 ชนิดตำหนิ
License: CC-BY-4.0

**ใช้ทำอะไร:** ทดสอบว่าโมเดล Stage 2 (เทรนบน NEU-DET + Roboflow rust/crack)
generalize ไปชุดเหล็กสายการผลิตอื่นได้แค่ไหน — โดยไม่ต้องถ่ายภาพเอง
(แทนแผน real_test 40-60 ภาพที่เก็บไม่ไหว)

**ไฟล์:**
- `images/`               229 ภาพ test split ของ GC10-DET
- `test_annotations.coco.json`   annotation COCO (จาก HF)
- `classmap.json`         map คลาส GC10 -> คลาสเรา (เฉพาะที่ตรงชัด: 7_yiwu = inclusion)

**รัน:**
```
python evaluate_cross_dataset.py --dir external_test/gc10 --map external_test/gc10/classmap.json
```
GC10 10 คลาส (pinyin): 1_chongkong เจาะรู, 2_hanfeng แนวเชื่อม, 3_yueyawan รอยเว้าเสี้ยว,
4_shuiban คราบน้ำ, 5_youban คราบน้ำมัน, 6_siban คราบไหม, 7_yiwu สิ่งเจือปน(inclusion),
8_yahen รอยกดจากการรีด, 9_zhehen รอยพับ, 10_yaozhed รอยงอเอว
