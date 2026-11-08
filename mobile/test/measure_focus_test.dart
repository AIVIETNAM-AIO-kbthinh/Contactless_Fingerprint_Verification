import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:fpsys_app/capture_screen.dart';

void main() {
  const w = 100, h = 80, stride = 104; // stride > width như plane Y thật
  Uint8List frame(int Function(int x, int y) f) {
    final b = Uint8List(stride * h);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        b[y * stride + x] = f(x, y);
      }
    }
    return b;
  }

  test('ảnh phẳng không nét, vân sọc nét', () {
    final flat = measureFocus(frame((x, y) => 120), w, h, stride);
    expect(flat.sharp, 0);
    expect(flat.bright, 120);
    // sọc chu kỳ 4 px (giống vân tay): chênh lệch cách 2 px là cực đại
    final ridges = measureFocus(frame((x, y) => (x ~/ 2).isEven ? 60 : 180), w, h, stride);
    expect(ridges.sharp, greaterThan(0.5));
  });

  test('màu da ở vùng giữa', () {
    // plane màu nửa độ phân giải, xen kẽ (bytesPerPixel = 2) như NV21 trên Android
    const uvStride = 104;
    Uint8List plane(int val) => Uint8List(uvStride * h ~/ 2)..fillRange(0, uvStride * h ~/ 2, val);
    final y = frame((x, y) => 150);
    final skin =
        measureFocus(y, w, h, stride, u: plane(120), v: plane(150), uvStride: uvStride, uvPixel: 2);
    expect(skin.skin, 1);
    final wall = measureFocus(y, w, h, stride,
        u: plane(128), v: plane(128), uvStride: uvStride, uvPixel: 2); // xám
    expect(wall.skin, 0);
    expect(measureFocus(y, w, h, stride).skin, 1); // không có plane màu
  });

  test('ảnh đen không chia cho 0', () {
    expect(measureFocus(frame((x, y) => 0), w, h, stride).sharp, 0);
  });
}
