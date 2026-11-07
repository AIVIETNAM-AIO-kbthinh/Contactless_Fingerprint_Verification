// Chụp đầu ngón: camera sau, độ phân giải tối đa, bật đèn, chạm để lấy nét.
// Trả về đường dẫn file JPEG gốc qua Navigator.pop (null nếu hủy).
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

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
    } catch (e) {
      setState(() => _error = 'Không mở được camera: $e');
    }
  }

  @override
  void dispose() {
    _c?.dispose();
    super.dispose();
  }

  void _focus(TapUpDetails d, BoxConstraints box) {
    final p = Offset(d.localPosition.dx / box.maxWidth, d.localPosition.dy / box.maxHeight);
    // một số máy không hỗ trợ chọn điểm lấy nét: bỏ qua lỗi
    _c!.setFocusPoint(p).catchError((_) {});
    _c!.setExposurePoint(p).catchError((_) {});
  }

  Future<void> _shoot() async {
    setState(() => _busy = true);
    try {
      final f = await _c!.takePicture();
      if (mounted) Navigator.pop(context, f.path);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Lỗi chụp: $e')));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = _c;
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
                              CustomPaint(painter: _OvalGuide()),
                            ]),
                          ),
                        ),
                      ),
                    ),
                  ),
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
  @override
  void paint(Canvas canvas, Size s) {
    final oval = Rect.fromCenter(
        center: s.center(Offset.zero), width: s.width * 0.5, height: s.height * 0.45);
    final dim = Path()
      ..addRect(Offset.zero & s)
      ..addOval(oval)
      ..fillType = PathFillType.evenOdd;
    canvas.drawPath(dim, Paint()..color = Colors.black45);
    canvas.drawOval(oval, Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
