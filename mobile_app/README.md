# steel_defect_scan

แอปมือถือ (Flutter) สำหรับ [steel-defect-detection](../../steel-defect-detection) — ต่อกับ
`mobile_api.py` (FastAPI) ที่รันบน PC แทนการฝังโมเดลลงเครื่อง ทำให้ได้ผลตรงกับหน้าเว็บ (`app.py`)
เป๊ะ ๆ (โมเดล/threshold ชุดเดียวกัน คือ `train-real3` ในปัจจุบัน)

ปรับจากต้นแบบ `app0821c` — คงโครง Android/iOS/gradle เดิมไว้ (build ได้แน่นอนกว่าสร้างโปรเจกต์ใหม่)
เปลี่ยนแค่ `pubspec.yaml` (ตัด `tflite_v2` ออก เพิ่ม `http` + `shared_preferences`) กับ `lib/main.dart`

## รันครั้งแรก

```powershell
cd D:\mob_app\steel_defect_scan
flutter pub get
flutter run          # เลือกอุปกรณ์ (LDPlayer/มือถือจริงที่ต่อ USB debug)
```

## ต้องรัน backend ก่อนเปิดแอป

ที่เครื่อง PC (ในโฟลเดอร์ `steel-defect-detection`):

```powershell
venv\Scripts\activate
python mobile_api.py                 # http://0.0.0.0:8000 — โชว์ IP วง LAN ให้ตอนเริ่ม
```

แล้วในแอป กดไอคอน ⚙️ ที่หน้าแรก ใส่ IP ของเครื่อง PC เช่น `http://192.168.1.102:8000`
(มือถือ/LDPlayer ต้องอยู่วง WiFi/เครือข่ายเดียวกับ PC — ตรวจ Windows Firewall อนุญาต inbound พอร์ต 8000 ด้วย)
กดบันทึกแล้วแอปจะ ping `/health` ให้ทันทีว่าเชื่อมได้ไหม

## 3 โหมดในแอป

| ปุ่ม | ทำอะไร | เรียก endpoint |
|---|---|---|
| **เลือกภาพ** | เลือกภาพจากคลังภาพ | `POST /detect` (pipeline เต็ม: Stage 1 หาพื้นที่เหล็ก + Stage 2) |
| **ถ่ายภาพ** | ถ่ายภาพเดี่ยวด้วยกล้อง | `POST /detect` |
| **เรียลไทม์** | สตรีมกล้อง ถ่ายเฟรมส่งตรวจทุก ~0.9 วิ วาดกรอบทับพรีวิวสด + แผงผล 8 ชนิดมุมบน | `POST /detect_fast` (ข้าม Stage 1 ให้เร็วพอดูสด) |

ผลตรวจแต่ละจุดแตะเพื่อดู "สาเหตุที่พบบ่อย" + "คำแนะนำ" (มาจาก `pipeline.DEFECT_INFO` ฝั่ง Python
ชุดเดียวกับกล่อง "สาเหตุ + คำแนะนำ" ในหน้าเว็บ)

## หมายเหตุ

- `AndroidManifest.xml` (`android/app/src/main/`) เพิ่ม `INTERNET` + `CAMERA` permission และ
  `android:usesCleartextTraffic="true"` ให้แล้ว (จำเป็น เพราะ backend เป็น `http://` ธรรมดา ไม่ใช่ https —
  Android 9+ บล็อก cleartext เป็นค่าเริ่มต้น)
- ยังไม่ได้เปลี่ยน Android `applicationId`/`namespace` (`com.example.app0821c` เดิม) — เพื่อไม่ให้เสี่ยง
  build พังจากการย้าย path ของ `MainActivity.kt` โดยไม่มี Flutter CLI ช่วยตรวจ ไม่กระทบการใช้งาน
- `lib/main.dart` เขียนใหม่ทั้งไฟล์ — ไม่มี TFLite/on-device model แล้ว ตรวจทุกอย่างผ่าน API
- `pubspec.lock` / `.flutter-plugins*` ถูกลบไว้ (มาจาก template เดิม ค้างอ้างอิง `tflite_v2`) —
  `flutter pub get` จะสร้างใหม่ให้ตรงกับ `pubspec.yaml` ปัจจุบัน
