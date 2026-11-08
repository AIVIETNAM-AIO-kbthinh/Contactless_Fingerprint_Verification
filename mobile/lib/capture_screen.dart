// Chụp đầu ngón: camera sau, độ phân giải tối đa, bật đèn, chạm để lấy nét.
// Tự quét: đo độ nét khung xem trước, đủ nét liên tục ~1 s thì tự chụp; nút Chụp để chụp tay.
// Trả về đường dẫn file JPEG gốc qua Navigator.pop (null nếu hủy).
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

// shortcut: ngưỡng đo trên 40 ảnh thật (data/test, thu về 1280×960); máy khác có thể cần chỉnh
// theo dòng chẩn đoán dưới khung xem
const _sharpMin = 0.017; // 37/40 ảnh thật đạt (nét 0.013–0.038); làm mờ 2 px còn trung vị ~0.014
const _skinMin = 0.7; // ảnh thật: 73–100% vùng giữa là màu da
// ngón được đèn chiếu ở 10–15 cm luôn sáng (ảnh thật 133–198); tường/nền thường tối hơn nhiều
const _brightMin = 110;
const _need = 5; // số lần đạt liên tiếp (mỗi lần cách ~200 ms) trước khi tự chụp

/// Đo vùng giữa 30%×30% của khung YUV420 (lấy mẫu cách 2 px):
/// - sharp: TB |chênh lệch ngang| + |chênh lệch dọc| của Y, chia độ sáng TB để không phụ thuộc đèn;
/// - bright: độ sáng TB;
/// - skin: tỉ lệ điểm màu da (Cr 135–180, Cb 85–135); không có plane màu thì coi là 1.
({double sharp, double bright, double skin}) measureFocus(Uint8List y, int w, int h, int stride,
    {Uint8List? u, Uint8List? v, int uvStride = 0, int uvPixel = 1}) {
  final x0 = (w * 0.35).toInt(), x1 = (w * 0.65).toInt();
  final y0 = (h * 0.35).toInt(), y1 = (h * 0.65).toInt();
  var grad = 0, sum = 0, n = 0, skin = 0;
  final color = u != null && v != null;
  for (var r = y0; r < y1 - 2; r += 2) {
    final row = r * stride;
    for (var c = x0; c < x1 - 2; c += 2) {
      final p = y[row + c];
      grad += (y[row + c + 2] - p).abs() + (y[row + 2 * stride + c] - p).abs();
      sum += p;
      n++;
      if (color) {
        final i = (r >> 1) * uvStride + (c >> 1) * uvPixel;
        final cb = u[i], cr = v[i];
        if (cr >= 135 && cr <= 180 && cb >= 85 && cb <= 135) skin++;
      }
    }
  }
  if (n == 0 || sum == 0) return (sharp: 0, bright: 0, skin: 0);
  final bright = sum / n;
  return (sharp: grad / n / bright, bright: bright, skin: color ? skin / n : 1);
}

class CaptureScreen extends StatefulWidget {
  final String title;
  const CaptureScreen({super.key, required this.title});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  CameraController? _c;
  String? _error;
  bool _busy = false;
  bool _scanning = false;
  int _streak = 0;
  int _lastMs = 0;
  double _lastSharp = 0;
  String _hint = 'Đưa đầu ngón vào khung, app sẽ tự chụp';
  String _diag = '';

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final cams = await availableCameras();
      final back = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => cams.first);
      final c = CameraController(back, ResolutionPreset.max, enableAudio: false);
      await c.initialize();
      // plugin giữ torchEnabled=true qua các lần mở camera nên bỏ qua lệnh bật: tắt trước để đặt lại
      await c.setFlashMode(FlashMode.off);
      await c.setFlashMode(FlashMode.torch);
      // camera chính chỉ lấy nét từ ~10 cm: zoom để ngón đủ to mà không phải đưa gần hơn
      await c.setZoomLevel(min(2.0, await c.getMaxZoomLevel()));
      if (!mounted) {
        await c.dispose();
        return;
      }
      setState(() => _c = c);
      await _startScan();
    } catch (e) {
      setState(() => _error = 'Không mở được camera: $e');
    }
  }

  @override
  void dispose() {
    _c?.dispose();
    super.dispose();
  }

  Future<void> _startScan() async {
    try {
      await _c!.startImageStream(_onFrame);
      if (mounted) setState(() => _scanning = true);
    } catch (e) {
      // máy không cho stream khung hình: chỉ chụp tay
      debugPrint('Không bật được tự quét: $e');
    }
  }

  void _onFrame(CameraImage img) {
    final now = DateTime.now().millisecondsSinceEpoch;
    if (_busy || !mounted || now - _lastMs < 200) return;
    _lastMs = now;
    final p = img.planes;
    final m = p.length >= 3
        ? measureFocus(p[0].bytes, img.width, img.height, p[0].bytesPerRow,
            u: p[1].bytes,
            v: p[2].bytes,
            uvStride: p[1].bytesPerRow,
            uvPixel: p[1].bytesPerPixel ?? 1)
        : measureFocus(p[0].bytes, img.width, img.height, p[0].bytesPerRow);
    // ổn định: độ nét gần như không đổi giữa 2 lần đo = tay giữ yên và đã lấy nét xong
    final steady = _lastSharp > 0 && (m.sharp - _lastSharp).abs() < 0.15 * _lastSharp;
    _lastSharp = m.sharp;
    final String hint;
    if (m.skin < _skinMin) {
      hint = 'Chưa thấy ngón: đưa đầu ngón che kín giữa khung';
    } else if (m.bright < _brightMin || m.bright > 235) {
      hint = m.bright < _brightMin
          ? 'Tối quá: đưa ngón gần đèn hơn'
          : 'Quá chói: lùi ngón ra xa một chút';
    } else if (m.sharp < _sharpMin) {
      hint = 'Chưa nét: đưa ngón xa/gần hơn hoặc chạm vào ngón để lấy nét';
    } else {
      hint = 'Giữ yên…';
    }
    final ok = hint == 'Giữ yên…' && steady;
    final pv = _c?.value.previewSize;
    setState(() {
      _streak = ok ? _streak + 1 : 0;
      _hint = hint;
      // shortcut: dòng chẩn đoán để hiệu chỉnh ngưỡng trên máy thật, bỏ khi đã chỉnh xong
      _diag = 'xem trước ${pv?.width.round()}×${pv?.height.round()} · '
          'phân tích ${img.width}×${img.height} · nét ${m.sharp.toStringAsFixed(3)} · '
          'da ${(m.skin * 100).round()}% · sáng ${m.bright.round()}';
    });
    if (_streak >= _need) _shoot();
  }

  Future<void> _shoot() async {
    if (_busy) return;
    // không dừng stream: plugin đã gắn sẵn phân tích cùng chụp ảnh, khung tới trong lúc chụp bị bỏ qua
    setState(() => _busy = true);
    try {
      final f = await _c!.takePicture();
      if (mounted) Navigator.pop(context, f.path);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Lỗi chụp: $e')));
      }
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          _streak = 0;
        });
      }
    }
  }

  void _focus(TapUpDetails d, BoxConstraints box) {
    final p = Offset(d.localPosition.dx / box.maxWidth, d.localPosition.dy / box.maxHeight);
    // một số máy không hỗ trợ chọn điểm lấy nét: bỏ qua lỗi
    _c!.setFocusPoint(p).catchError((_) {});
    _c!.setExposurePoint(p).catchError((_) {});
  }

  @override
  Widget build(BuildContext context) {
    final c = _c;
    final color = _busy
        ? Colors.greenAccent
        : _streak > 0
            ? Colors.amber
            : Colors.white;
    final status = _busy
        ? 'Đang chụp…'
        : _scanning
            ? _hint
            : 'Bấm Chụp khi ngón đã nét';
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: _error != null
          ? Center(child: Padding(padding: const EdgeInsets.all(16), child: Text(_error!)))
          : c == null
              ? const Center(child: CircularProgressIndicator())
              : Column(children: [
                  Expanded(
                    child: Center(
                      child: AspectRatio(
                        aspectRatio: 1 / c.value.aspectRatio, // máy dọc
                        child: LayoutBuilder(
                          builder: (ctx, box) => GestureDetector(
                            onTapUp: (d) => _focus(d, box),
                            child: Stack(fit: StackFit.expand, children: [
                              CameraPreview(c),
                              CustomPaint(painter: _OvalGuide(color, _busy ? 1 : _streak / _need)),
                            ]),
                          ),
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
                    child: Text(status,
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.titleMedium),
                  ),
                  if (_diag.isNotEmpty)
                    Text(_diag,
                        textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
                  const Padding(
                    padding: EdgeInsets.all(12),
                    child: Text(
                      'Một ngón, mặt bụng ngón hướng vào camera · cách 10–15 cm · '
                      'chạm vào đầu ngón trên màn hình để lấy nét',
                      textAlign: TextAlign.center,
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(bottom: 24),
                    child: FilledButton.icon(
                      onPressed: _busy ? null : _shoot,
                      icon: const Icon(Icons.camera_alt),
                      label: Text(_busy ? 'Đang chụp…' : 'Chụp'),
                    ),
                  ),
                ]),
    );
  }
}

class _OvalGuide extends CustomPainter {
  final Color color;
  final double progress; // 0..1: tiến độ giữ yên trước khi tự chụp
  _OvalGuide(this.color, this.progress);

  @override
  void paint(Canvas canvas, Size s) {
    final oval = Rect.fromCenter(
        center: s.center(Offset.zero), width: s.width * 0.5, height: s.height * 0.45);
    final dim = Path()
      ..addRect(Offset.zero & s)
      ..addOval(oval)
      ..fillType = PathFillType.evenOdd;
    canvas.drawPath(dim, Paint()..color = Colors.black45);
    canvas.drawOval(
        oval,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3);
    if (progress > 0) {
      canvas.drawArc(
          oval.inflate(8),
          -pi / 2,
          2 * pi * progress,
          false,
          Paint()
            ..color = color
            ..style = PaintingStyle.stroke
            ..strokeWidth = 6
            ..strokeCap = StrokeCap.round);
    }
  }

  @override
  bool shouldRepaint(_OvalGuide old) => old.color != color || old.progress != progress;
}
