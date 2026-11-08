// Gọi backend fpsys (server/app.py). Lỗi trả về câu tiếng Việt do server soạn sẵn.
import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

// Khớp FINGERS trong demo/session.py
const fingers = {
  'R1': 'Ngón cái phải',
  'R2': 'Ngón trỏ phải',
  'R3': 'Ngón giữa phải',
  'R4': 'Ngón áp út phải',
  'R5': 'Ngón út phải',
  'L1': 'Ngón cái trái',
  'L2': 'Ngón trỏ trái',
  'L3': 'Ngón giữa trái',
  'L4': 'Ngón áp út trái',
  'L5': 'Ngón út trái',
};

// mỗi mẫu một tư thế: mẫu đã lưu bao phủ nhiều góc nên lúc xác thực không phải căn đúng một góc
const enrollPoses = [
  'Để thẳng, bụng ngón nhìn thẳng camera',
  'Nghiêng nhẹ ngón sang trái',
  'Nghiêng nhẹ ngón sang phải',
  'Đầu ngón hơi ngửa lên',
  'Đầu ngón hơi cúi xuống',
];
final enrollSamples = enrollPoses.length;

// Gắn lúc build APK: --dart-define=SERVER_URL=https://... --dart-define=API_KEY=...
const _defaultUrl = String.fromEnvironment('SERVER_URL', defaultValue: 'http://192.168.1.5:8000');
const _defaultKey = String.fromEnvironment('API_KEY');

class ApiError implements Exception {
  final String message;
  final String? code; // ErrorCode của fpsys khi server trả 422
  ApiError(this.message, [this.code]);
  @override
  String toString() => message;
}

class Api {
  final String baseUrl;
  final String apiKey;
  Api(this.baseUrl, this.apiKey);

  static Future<Api> load() async {
    final p = await SharedPreferences.getInstance();
    // ô để trống trong Cài đặt nghĩa là dùng giá trị gắn lúc build
    String pick(String k, String def) {
      final v = p.getString(k) ?? '';
      return v.isEmpty ? def : v;
    }

    return Api(pick('url', _defaultUrl), pick('key', _defaultKey));
  }

  static Future<void> save(String url, String key) async {
    final p = await SharedPreferences.getInstance();
    await p.setString('url', url.trim().replaceAll(RegExp(r'/+$'), ''));
    await p.setString('key', key.trim());
  }

  Uri _u(String path) => Uri.parse('$baseUrl$path');

  String _finger(String name, String finger) =>
      '/users/${Uri.encodeComponent(name)}/fingers/$finger';

  Future<dynamic> _send(http.BaseRequest req) async {
    req.headers['X-API-Key'] = apiKey;
    final http.Response r;
    try {
      // extract() trên CPU mất vài giây mỗi ảnh
      r = await http.Response.fromStream(await req.send().timeout(const Duration(seconds: 90)));
    } catch (e) {
      throw ApiError('Không kết nối được server ($baseUrl). Kiểm tra Wi-Fi và địa chỉ.');
    }
    final body = r.bodyBytes.isEmpty ? null : jsonDecode(utf8.decode(r.bodyBytes));
    if (r.statusCode >= 400) {
      final msg = body is Map ? (body['message'] ?? body['detail']) : null;
      throw ApiError('${msg ?? 'Lỗi server ${r.statusCode}'}', body is Map ? body['code'] : null);
    }
    return body;
  }

  Future<http.MultipartRequest> _upload(String path, String field, List<String> files) async {
    final req = http.MultipartRequest('POST', _u(path));
    for (final f in files) {
      req.files.add(await http.MultipartFile.fromPath(field, f));
    }
    return req;
  }

  /// {ok, n_minutiae} hoặc {ok: false, code, message}
  Future<Map> check(String file) async => await _send(await _upload('/check', 'image', [file]));

  /// {kept, total}
  Future<Map> enroll(String name, String finger, List<String> files) async =>
      await _send(await _upload('${_finger(name, finger)}/enroll', 'images', files));

  /// {score, threshold, accept}
  Future<Map> verify(String name, String finger, String file) async =>
      await _send(await _upload('${_finger(name, finger)}/verify', 'image', [file]));

  /// [{name, fingers: [...]}]
  Future<List> users() async => await _send(http.Request('GET', _u('/users')));

  Future<void> deleteUser(String name) async =>
      await _send(http.Request('DELETE', _u('/users/${Uri.encodeComponent(name)}')));
}
