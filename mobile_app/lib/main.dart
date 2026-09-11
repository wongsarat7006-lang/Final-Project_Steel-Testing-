// steel_defect_scan — แอปมือถือสำหรับ steel-defect-detection
// เรียกตรวจผ่าน REST API (mobile_api.py, FastAPI) แทนการรันโมเดลบนเครื่อง
// ตั้งค่า "ที่อยู่เซิร์ฟเวอร์" ในหน้าแรก (ไอคอนฟันเฟือง) ให้ตรงกับเครื่องที่รัน:
//     python mobile_api.py
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ApiClient.load();
  try {
    cameras = await availableCameras();
  } catch (_) {
    cameras = [];
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'ตรวจตำหนิพื้นผิวเหล็ก',
        theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.blueGrey),
        home: const HomePage(),
      );
}

// ---------- สีตามระดับความเสี่ยง (ให้ตรงกับ pipeline.DEFECT_INFO ฝั่ง Python) ----------
const Map<String, Color> kRiskColor = {
  'สูง': Color(0xffef4444),
  'ปานกลาง-สูง': Color(0xfff97316),
  'ปานกลาง': Color(0xfff59e0b),
  'ต่ำ-ปานกลาง': Color(0xffeab308),
  'ต่ำ': Color(0xff22c55e),
};

Color riskColor(String risk) => kRiskColor[risk] ?? Colors.grey;

// int.clamp() คืน num ไม่ใช่ int (SDK) — ใช้ตัวนี้แทนกันพลาดตอนส่งเข้าพารามิเตอร์ที่รับ int ตรง ๆ
int _clamp8(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

// ---------- โมเดลข้อมูลผลตรวจ 1 จุด ----------
class Detection {
  final String cls, nameTh, risk, causes, advice;
  final double confidence;
  final List<double> bbox; // [x1, y1, x2, y2] พิกัด px ของภาพที่ส่งไปตรวจ

  Detection.fromJson(Map<String, dynamic> j)
      : cls = j['class'] as String,
        nameTh = j['name_th'] as String,
        risk = j['risk'] as String,
        causes = (j['causes'] ?? '') as String,
        advice = (j['advice'] ?? '') as String,
        confidence = (j['confidence'] as num).toDouble(),
        bbox = (j['bbox'] as List).map((e) => (e as num).toDouble()).toList();
}

// ---------- เรียก API ของ mobile_api.py ----------
class ApiClient {
  static String baseUrl = 'http://192.168.1.102:8000';

  static Future<void> load() async {
    final p = await SharedPreferences.getInstance();
    baseUrl = p.getString('base_url') ?? baseUrl;
  }

  static Future<void> save(String url) async {
    baseUrl = url;
    final p = await SharedPreferences.getInstance();
    await p.setString('base_url', url);
  }

  static Future<Map<String, dynamic>> _postBytes(
      List<int> bytes, String filename, bool fast) async {
    final uri = Uri.parse('$baseUrl${fast ? '/detect_fast' : '/detect'}');
    final req = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    final res = await req.send().timeout(const Duration(seconds: 20));
    final body = await res.stream.bytesToString();
    if (res.statusCode != 200) {
      throw Exception('เซิร์ฟเวอร์ตอบผิดพลาด (${res.statusCode}): $body');
    }
    return jsonDecode(body) as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>> detectFile(File file, {bool fast = false}) =>
      _postBytes(file.readAsBytesSync(), file.path.split(Platform.pathSeparator).last, fast);

  static Future<Map<String, dynamic>> detectBytes(List<int> bytes, {bool fast = true}) =>
      _postBytes(bytes, 'frame.jpg', fast);

  static Future<bool> ping() async {
    try {
      final r = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 4));
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}

// ================= หน้าแรก =================
class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final picker = ImagePicker();
  File? _file;
  List<Detection>? _dets;
  String? _error;
  bool _loading = false;

  Future<void> _pick(ImageSource source) async {
    final img = await picker.pickImage(source: source, imageQuality: 90);
    if (img == null) return;
    setState(() {
      _file = File(img.path);
      _dets = null;
      _error = null;
      _loading = true;
    });
    try {
      final json = await ApiClient.detectFile(_file!);
      final list = (json['detections'] as List)
          .map((e) => Detection.fromJson(e as Map<String, dynamic>))
          .toList();
      setState(() => _dets = list);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _openSettings() async {
    final ctrl = TextEditingController(text: ApiClient.baseUrl);
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('ที่อยู่เซิร์ฟเวอร์'),
        content: TextField(
          controller: ctrl,
          decoration: const InputDecoration(
              hintText: 'http://192.168.1.102:8000',
              helperText: 'รัน python mobile_api.py บน PC แล้วใส่ IP ของเครื่องนั้น'),
          keyboardType: TextInputType.url,
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('ยกเลิก')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('บันทึก')),
        ],
      ),
    );
    if (ok == true) {
      await ApiClient.save(ctrl.text.trim());
      final good = await ApiClient.ping();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(good ? 'เชื่อมต่อเซิร์ฟเวอร์สำเร็จ' : 'เชื่อมต่อไม่ได้ — เช็ก IP/พอร์ต/ไฟร์วอลล์'),
        backgroundColor: good ? Colors.green : Colors.red,
      ));
    }
  }

  void _openLive() {
    if (cameras.isEmpty) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('ไม่พบกล้องบนเครื่องนี้')));
      return;
    }
    Navigator.push(context, MaterialPageRoute(builder: (_) => const LiveScanPage()));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ตรวจตำหนิพื้นผิวเหล็ก'),
        actions: [IconButton(onPressed: _openSettings, icon: const Icon(Icons.settings))],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('เซิร์ฟเวอร์: ${ApiClient.baseUrl}',
                  style: Theme.of(context).textTheme.bodySmall, textAlign: TextAlign.center),
              const SizedBox(height: 12),
              if (_file != null)
                AspectRatio(
                  aspectRatio: 4 / 3,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: BoxOverlay(file: _file!, detections: _dets ?? const []),
                  ),
                ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _pick(ImageSource.gallery),
                      icon: const Icon(Icons.photo),
                      label: const Text('เลือกภาพ'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _pick(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('ถ่ายภาพ'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: _openLive,
                      icon: const Icon(Icons.videocam),
                      label: const Text('เรียลไทม์'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              if (_loading) const Center(child: CircularProgressIndicator()),
              if (_error != null)
                Card(
                  color: Colors.red.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Text('ตรวจไม่สำเร็จ: $_error', style: const TextStyle(color: Colors.red)),
                  ),
                ),
              if (_dets != null) ResultList(detections: _dets!),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------- รายการผลตรวจ + คำอธิบายสาเหตุ/คำแนะนำ (พับได้) ----------
class ResultList extends StatelessWidget {
  final List<Detection> detections;
  const ResultList({super.key, required this.detections});

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text('ไม่พบตำหนิในภาพนี้ (ถ้าเป็นภาพสไตล์ที่โมเดลไม่คุ้น อาจพลาดได้)'),
        ),
      );
    }
    return Column(
      children: detections
          .map((d) => Card(
                margin: const EdgeInsets.only(bottom: 8),
                child: ExpansionTile(
                  leading: CircleAvatar(
                    backgroundColor: riskColor(d.risk),
                    child: Text('${(d.confidence * 100).round()}',
                        style: const TextStyle(fontSize: 11, color: Colors.white)),
                  ),
                  title: Text(d.nameTh, style: const TextStyle(fontWeight: FontWeight.bold)),
                  subtitle: Text('ความเสี่ยง: ${d.risk} · มั่นใจ ${(d.confidence * 100).toStringAsFixed(0)}%'),
                  children: [
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          if (d.causes.isNotEmpty) Text('สาเหตุที่พบบ่อย: ${d.causes}'),
                          if (d.advice.isNotEmpty)
                            Padding(
                              padding: const EdgeInsets.only(top: 4),
                              child: Text('คำแนะนำ: ${d.advice}'),
                            ),
                        ],
                      ),
                    ),
                  ],
                ),
              ))
          .toList(),
    );
  }
}

// ---------- ภาพนิ่ง + วาดกรอบตำหนิทับ ----------
class BoxOverlay extends StatelessWidget {
  final File file;
  final List<Detection> detections;
  const BoxOverlay({super.key, required this.file, required this.detections});

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<ui.Image>(
      future: _decode(file),
      builder: (context, snap) {
        if (!snap.hasData) {
          return Image.file(file, fit: BoxFit.contain);
        }
        return CustomPaint(
          painter: _BoxPainter(image: snap.data!, detections: detections),
          child: Container(),
        );
      },
    );
  }

  Future<ui.Image> _decode(File f) async {
    final bytes = await f.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }
}

class _BoxPainter extends CustomPainter {
  final ui.Image image;
  final List<Detection> detections;
  _BoxPainter({required this.image, required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    // วาดภาพแบบ contain (คงสัดส่วน) ตรงกลาง แล้ว map พิกัดกรอบตามสเกลเดียวกัน
    final iw = image.width.toDouble(), ih = image.height.toDouble();
    final scale = (size.width / iw < size.height / ih) ? size.width / iw : size.height / ih;
    final dx = (size.width - iw * scale) / 2, dy = (size.height - ih * scale) / 2;
    final dst = Rect.fromLTWH(dx, dy, iw * scale, ih * scale);
    canvas.drawImageRect(image, Rect.fromLTWH(0, 0, iw, ih), dst, Paint());

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = const Color(0xffef4444);
    for (final d in detections) {
      final r = Rect.fromLTRB(
        dx + d.bbox[0] * scale, dy + d.bbox[1] * scale,
        dx + d.bbox[2] * scale, dy + d.bbox[3] * scale,
      );
      canvas.drawRect(r, boxPaint);
      final tp = TextPainter(
        text: TextSpan(
          text: '${d.nameTh} ${(d.confidence * 100).round()}%',
          style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.bold),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final labelY = r.top - tp.height - 2 >= 0 ? r.top - tp.height - 2 : r.top + 2;
      canvas.drawRect(Rect.fromLTWH(r.left, labelY, tp.width + 8, tp.height + 3),
          Paint()..color = Colors.black.withOpacity(0.72));
      tp.paint(canvas, Offset(r.left + 4, labelY + 1));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) =>
      old.image != image || old.detections != detections;
}

// ================= โหมดเรียลไทม์ (กล้องสด) =================
class LiveScanPage extends StatefulWidget {
  const LiveScanPage({super.key});
  @override
  State<LiveScanPage> createState() => _LiveScanPageState();
}

class _LiveScanPageState extends State<LiveScanPage> {
  late CameraController _controller;
  bool _ready = false, _busy = false, _streaming = false;
  List<Detection> _dets = [];
  Size? _lastImageSize;
  String? _error;
  DateTime _lastSent = DateTime.fromMillisecondsSinceEpoch(0);
  int _okCount = 0, _errCount = 0;

  static const List<String> _allClasses = [
    'รอยแตกลายงา', 'สิ่งแปลกปลอมฝังใน', 'รอยแผ่น/ผิวลอก', 'ผิวขรุขระเป็นหลุม',
    'สะเก็ดฝังจากการรีด', 'รอยขีดข่วน', 'สนิม', 'รอยแตกร้าว',
  ];

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    // medium: คมชัดพอตรวจตำหนิละเอียด ๆ ได้ (low เดิมภาพเบลอเกินไป) ยังแปลง+ส่งทันภายใน ~0.9 วิ
    _controller = CameraController(cameras[0], ResolutionPreset.medium,
        enableAudio: false, imageFormatGroup: ImageFormatGroup.yuv420);
    await _controller.initialize();
    if (!mounted) return;
    setState(() => _ready = true);
    await _controller.startImageStream(_onFrame);
    _streaming = true;
  }

  // เรียกทุกเฟรมจากกล้อง (~15-30 fps) — ตัดทิ้งส่วนใหญ่ ส่งจริงแค่ทุก ~900ms
  // และห้ามซ้อนคำขอ (ถ้าคำขอก่อนหน้ายังไม่เสร็จ ข้ามเฟรมนี้ไปเลย)
  void _onFrame(CameraImage image) {
    if (_busy) return;
    if (DateTime.now().difference(_lastSent).inMilliseconds < 900) return;
    _busy = true;
    _lastSent = DateTime.now();
    _processFrame(image);
  }

  Future<void> _processFrame(CameraImage image) async {
    try {
      final jpg = _yuv420ToJpeg(image);
      final json = await ApiClient.detectBytes(jpg, fast: true);
      final list = (json['detections'] as List)
          .map((e) => Detection.fromJson(e as Map<String, dynamic>))
          .toList();
      _okCount++;
      if (mounted) {
        setState(() {
          _dets = list;
          _lastImageSize = Size(image.width.toDouble(), image.height.toDouble());
          _error = null;
        });
      }
    } catch (e) {
      _errCount++;
      if (mounted) setState(() => _error = 'เชื่อมต่อเซิร์ฟเวอร์ไม่ได้: $e');
    } finally {
      _busy = false;
    }
  }

  // YUV420 (3 plane: Y, U, V — ตามที่ camera plugin ให้บน Android) -> JPEG
  // แปลงบน isolate แยกจะลื่นกว่านี้ได้อีก แต่ที่ ResolutionPreset.low ก็เร็วพอสำหรับทุก ~0.9 วิ
  Uint8List _yuv420ToJpeg(CameraImage image) {
    final w = image.width, h = image.height;
    final yPlane = image.planes[0], uPlane = image.planes[1], vPlane = image.planes[2];
    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;
    final out = img.Image(width: w, height: h);

    for (int y = 0; y < h; y++) {
      final yRow = y * yRowStride;
      final uvRow = (y >> 1) * uvRowStride;
      for (int x = 0; x < w; x++) {
        final yVal = yPlane.bytes[yRow + x];
        final uvCol = (x >> 1) * uvPixelStride;
        final uVal = uPlane.bytes[uvRow + uvCol];
        final vVal = vPlane.bytes[uvRow + uvCol];
        final c = yVal - 16, d = uVal - 128, e = vVal - 128;
        final r = (298 * c + 409 * e + 128) >> 8;
        final g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        final b = (298 * c + 516 * d + 128) >> 8;
        out.setPixelRgb(x, y, _clamp8(r), _clamp8(g), _clamp8(b));
      }
    }
    return Uint8List.fromList(img.encodeJpg(out, quality: 78));
  }

  @override
  void dispose() {
    if (_streaming) _controller.stopImageStream();
    if (_ready) _controller.dispose();
    super.dispose();
  }

  // การ์ดสาเหตุ/คำแนะนำของตำหนิที่แตะ — ดึงจาก pipeline.DEFECT_INFO ฝั่ง Python
  // (โปรเจกต์นี้ทำเพื่อ "ตรวจจับ + อธิบายได้" ไม่ใช่แค่ทายชื่อชนิด)
  void _showDetail(BuildContext context, Detection d) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(16))),
      builder: (_) => Padding(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 28),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                    backgroundColor: riskColor(d.risk),
                    child: Text('${(d.confidence * 100).round()}',
                        style: const TextStyle(fontSize: 11, color: Colors.white))),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(d.nameTh, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 17)),
                      Text('ความเสี่ยง: ${d.risk}', style: const TextStyle(color: Colors.black54)),
                    ],
                  ),
                ),
              ],
            ),
            const Divider(height: 24),
            if (d.causes.isNotEmpty) ...[
              const Text('สาเหตุที่พบบ่อย', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              Text(d.causes),
              const SizedBox(height: 14),
            ],
            if (d.advice.isNotEmpty) ...[
              const Text('คำแนะนำ', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              Text(d.advice),
            ],
            const SizedBox(height: 8),
            const Text('ข้อมูลอ้างอิงทั่วไปของตำหนิชนิดนี้ — ไม่ใช่การวินิจฉัยชิ้นงานที่กำลังส่องอยู่',
                style: TextStyle(color: Colors.black45, fontSize: 12)),
          ],
        ),
      ),
    );
  }

  // เซลล์เดียวในแผง 2 คอลัมน์: ชื่อคลาส + % ชิดขวา
  Widget _classCell(String name, Map<String, double> found, String? top) {
    final v = found[name] ?? 0;
    final on = name == top && v > 0;
    final color = on ? Colors.cyanAccent : Colors.white70;
    return Row(
      children: [
        Expanded(
          child: Text(name,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                  color: color, fontWeight: on ? FontWeight.bold : FontWeight.normal, fontSize: 11.5)),
        ),
        Text(v > 0 ? '${(v * 100).toStringAsFixed(0)}%' : '—',
            style: TextStyle(
              color: on ? Colors.cyanAccent : Colors.white38,
              fontWeight: on ? FontWeight.bold : FontWeight.normal,
              fontSize: 11.5,
              fontFeatures: const [ui.FontFeature.tabularFigures()],
            )),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator(color: Colors.white)),
      );
    }
    final found = <String, double>{};
    for (final d in _dets) {
      found[d.nameTh] = (found[d.nameTh] ?? 0) > d.confidence ? found[d.nameTh]! : d.confidence;
    }
    String? top;
    double mx = 0;
    for (final k in _allClasses) {
      final v = found[k] ?? 0;
      if (v > mx) { mx = v; top = k; }
    }

    final connected = _error == null;
    final topDet = _dets.isEmpty
        ? null
        : (_dets.toList()..sort((a, b) => b.confidence.compareTo(a.confidence))).first;
    final bannerColor = _error != null
        ? Colors.red.shade800
        : (topDet == null ? const Color(0xff16a34a) : riskColor(topDet.risk));
    final bannerText = _error != null
        ? 'เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ (ลองใหม่ทุก ~1 วิ)'
        : (topDet == null
            ? '✓  ยังไม่พบตำหนิ'
            : '⚠  ${topDet.nameTh} ${(topDet.confidence * 100).round()}%'
                '${_dets.length > 1 ? '  (+${_dets.length - 1} จุดอื่น)' : ''}');

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          Positioned.fill(child: CameraPreview(_controller)),
          if (_lastImageSize != null)
            Positioned.fill(
              child: CustomPaint(
                painter: _LiveBoxPainter(imageSize: _lastImageSize!, detections: _dets),
              ),
            ),
          // แถบหัวข้อ — ปุ่มย้อนกลับ + ชื่อโหมด + จุดสถานะเชื่อมต่อ
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.fromLTRB(4, MediaQuery.of(context).padding.top + 4, 14, 10),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter, end: Alignment.bottomCenter,
                  colors: [Colors.black.withOpacity(0.75), Colors.black.withOpacity(0)],
                ),
              ),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back, color: Colors.white),
                    onPressed: () => Navigator.pop(context),
                  ),
                  const Text('ตรวจแบบเรียลไทม์',
                      style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15)),
                  const Spacer(),
                  Container(
                    width: 8, height: 8,
                    decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: connected ? const Color(0xff22c55e) : const Color(0xffef4444)),
                  ),
                  const SizedBox(width: 6),
                  Text('$_okCount/${_okCount + _errCount} ครั้ง',
                      style: const TextStyle(color: Colors.white70, fontSize: 11)),
                ],
              ),
            ),
          ),
          // แผงไล่ 8 ชนิด — 2 คอลัมน์ ให้เตี้ยลงครึ่งหนึ่ง จะได้ไม่บังกล้องเยอะเกินไป
          Positioned(
            top: MediaQuery.of(context).padding.top + 52,
            left: 12,
            right: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.55),
                  borderRadius: BorderRadius.circular(12)),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  for (int i = 0; i < _allClasses.length; i += 2)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 1.5),
                      child: Row(
                        children: [
                          Expanded(child: _classCell(_allClasses[i], found, top)),
                          const SizedBox(width: 10),
                          Expanded(child: _classCell(_allClasses[i + 1], found, top)),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          ),
          // แถบผลสรุปด้านล่าง — สีตามความเสี่ยง/สถานะเชื่อมต่อ · แตะเพื่อดูสาเหตุ+คำแนะนำ
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: Material(
              color: bannerColor,
              child: InkWell(
                onTap: topDet == null ? null : () => _showDetail(context, topDet),
                child: Padding(
                  padding: EdgeInsets.fromLTRB(16, 14, 16, MediaQuery.of(context).padding.bottom + 14),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(bannerText,
                          textAlign: TextAlign.center,
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15)),
                      if (topDet != null)
                        const Padding(
                          padding: EdgeInsets.only(top: 3),
                          child: Text('แตะเพื่อดูสาเหตุ + คำแนะนำ',
                              style: TextStyle(color: Colors.white70, fontSize: 11.5)),
                        ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _LiveBoxPainter extends CustomPainter {
  final Size imageSize;
  final List<Detection> detections;
  _LiveBoxPainter({required this.imageSize, required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    final scale = (size.width / imageSize.width < size.height / imageSize.height)
        ? size.width / imageSize.width
        : size.height / imageSize.height;
    final dx = (size.width - imageSize.width * scale) / 2;
    final dy = (size.height - imageSize.height * scale) / 2;
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = const Color(0xffef4444);
    for (final d in detections) {
      final r = Rect.fromLTRB(
        dx + d.bbox[0] * scale, dy + d.bbox[1] * scale,
        dx + d.bbox[2] * scale, dy + d.bbox[3] * scale,
      );
      canvas.drawRect(r, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _LiveBoxPainter old) =>
      old.imageSize != imageSize || old.detections != detections;
}
