// Đăng ký: nhập tên, chọn ngón, chụp enrollSamples tấm (mỗi tấm kiểm tra /check ngay).
import 'package:flutter/material.dart';

import 'api.dart';
import 'capture_screen.dart';
import 'theme.dart';

class EnrollScreen extends StatefulWidget {
  const EnrollScreen({super.key});

  @override
  State<EnrollScreen> createState() => _EnrollScreenState();
}

class _EnrollScreenState extends State<EnrollScreen> {
  final _name = TextEditingController();
  String _finger = 'R2';
  final List<String> _shots = [];
  String? _status;
  bool _busy = false;

  Future<void> _run(Future<void> Function(Api api) job) async {
    setState(() => _busy = true);
    try {
      await job(await Api.load());
    } on ApiError catch (e) {
      setState(() => _status = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _capture() async {
    final i = _shots.length;
    final path = await Navigator.push<String>(
        context,
        MaterialPageRoute(
            builder: (_) =>
                CaptureScreen(title: 'Mẫu ${i + 1}/$enrollSamples', pose: enrollPoses[i])));
    if (path == null) return;
    var next = false;
    await _run((api) async {
      setState(() => _status = 'Đang kiểm tra ảnh…');
      final r = await api.check(path);
      setState(() {
        if (r['ok'] == true) {
          _shots.add(path);
          _status = 'Mẫu ${_shots.length} đạt (${r['n_minutiae']} điểm đặc trưng).';
          next = _shots.length < enrollSamples;
        } else {
          _status = '${r['message']} Hãy chụp lại.';
        }
      });
    });
    // mẫu đạt: mở luôn màn chụp mẫu tiếp; mẫu hỏng thì dừng để người dùng đọc lý do
    if (next && mounted) await _capture();
  }

  Future<void> _enroll() async {
    setState(() {
      _busy = true;
      _status = 'Đang đăng ký…';
    });
    try {
      final r = await (await Api.load()).enroll(_name.text.trim(), _finger, _shots);
      if (!mounted) return;
      await showDialog(
          context: context,
          builder: (ctx) => AlertDialog(
                title: const Text('Đăng ký thành công'),
                content: Text('Đã lưu ${r['kept']}/${r['total']} mẫu cho ${_name.text.trim()}.'),
                actions: [TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('OK'))],
              ));
      if (mounted) Navigator.pop(context);
    } on ApiError catch (e) {
      setState(() {
        _status = e.message;
        if (e.code == 'INCONSISTENT_ENROLL') _shots.clear(); // chụp lại từ đầu
      });
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final named = _name.text.trim().isNotEmpty;
    final full = _shots.length >= enrollSamples;
    return Scaffold(
      appBar: AppBar(title: const Text('Đăng ký vân tay')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(children: [
              TextField(
                controller: _name,
                decoration: const InputDecoration(labelText: 'Họ tên'),
                onChanged: (_) => setState(() {}),
              ),
              const SizedBox(height: 20),
              DropdownButtonFormField<String>(
                initialValue: _finger,
                decoration: const InputDecoration(labelText: 'Ngón'),
                items: [
                  for (final e in fingers.entries)
                    DropdownMenuItem(value: e.key, child: Text('${e.key} · ${e.value}')),
                ],
                onChanged: _busy
                    ? null
                    : (v) => setState(() {
                          _finger = v!;
                          _shots.clear(); // mẫu cũ là của ngón khác
                        }),
              ),
            ]),
          ),
        ),
        const SizedBox(height: 28),
        _Steps(done: _shots.length),
        const SizedBox(height: 10),
        Text('Đã chụp ${_shots.length}/$enrollSamples mẫu đạt',
            textAlign: TextAlign.center, style: const TextStyle(color: kMuted)),
        const SizedBox(height: 24),
        if (!full)
          FilledButton.icon(
            onPressed: _busy || !named ? null : _capture,
            icon: const Icon(Icons.fingerprint),
            label: Text('Quét mẫu ${_shots.length + 1}'),
          )
        else
          FilledButton.icon(
            onPressed: _busy || !named ? null : _enroll,
            icon: const Icon(Icons.check),
            label: const Text('Hoàn tất đăng ký'),
          ),
        if (_busy)
          const Padding(
              padding: EdgeInsets.all(16), child: Center(child: CircularProgressIndicator())),
        if (_status != null)
          Padding(padding: const EdgeInsets.only(top: 16), child: StatusNote(_status!)),
      ]),
    );
  }
}

/// Tiến độ dạng các bước tròn nối nhau; bước đã đạt có dấu ✓.
class _Steps extends StatelessWidget {
  final int done;
  const _Steps({required this.done});

  Widget _dot(int i) {
    final ok = i < done, now = i == done;
    return Container(
      width: 44,
      height: 44,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: ok ? kAccent : kCard,
        border: Border.all(color: ok || now ? kAccent : kField, width: 2),
      ),
      child: ok
          ? const Icon(Icons.check, color: Color(0xFF04201C))
          : Text('${i + 1}', style: TextStyle(color: now ? kAccent : kMuted)),
    );
  }

  @override
  Widget build(BuildContext context) => Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        for (var i = 0; i < enrollSamples; i++) ...[
          if (i > 0) Container(width: 16, height: 2, color: i <= done ? kAccent : kField),
          _dot(i),
        ],
      ]);
}
