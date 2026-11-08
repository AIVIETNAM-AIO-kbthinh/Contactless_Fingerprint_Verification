// Xác thực 1:1: chọn người + ngón đã đăng ký, chụp 1 tấm, hiện CHẤP NHẬN / TỪ CHỐI.
import 'package:flutter/material.dart';

import 'api.dart';
import 'capture_screen.dart';
import 'theme.dart';

const _maxTries = 3;

class VerifyScreen extends StatefulWidget {
  const VerifyScreen({super.key});

  @override
  State<VerifyScreen> createState() => _VerifyScreenState();
}

class _VerifyScreenState extends State<VerifyScreen> {
  List<Map> _users = [];
  String? _name;
  String? _finger;
  Map? _result;
  String? _status;
  bool _busy = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final us = (await (await Api.load()).users())
          .cast<Map>()
          .where((u) => (u['fingers'] as List).isNotEmpty)
          .toList();
      setState(() {
        _users = us;
        _status = us.isEmpty ? 'Chưa có ai đăng ký.' : null;
      });
    } on ApiError catch (e) {
      setState(() => _status = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  List<String> get _fingersOf => _name == null
      ? []
      : (_users.firstWhere((u) => u['name'] == _name)['fingers'] as List).cast<String>();

  /// Tự mở lại màn quét khi chưa khớp hoặc ảnh chưa đạt, tối đa [_maxTries] lần.
  Future<void> _verify() async {
    setState(() {
      _result = null;
      _status = null;
    });
    for (var i = 1; i <= _maxTries; i++) {
      final path = await Navigator.push<String>(
          context,
          MaterialPageRoute(
              builder: (_) =>
                  CaptureScreen(title: i == 1 ? 'Xác thực' : 'Xác thực · lần $i/$_maxTries')));
      if (path == null || !mounted) return;
      setState(() {
        _busy = true;
        _status = 'Đang xác thực…';
      });
      try {
        final r = await (await Api.load()).verify(_name!, _finger!, path);
        final accept = r['accept'] == true;
        if (accept || i == _maxTries) {
          setState(() {
            _result = r;
            _status = null;
          });
          return;
        }
      } on ApiError catch (e) {
        setState(() => _status = e.message);
        // chỉ thử lại khi lỗi do ảnh; lỗi mạng hoặc mẫu không tương thích thì dừng
        if (e.code == null || e.code == 'TEMPLATE_MISMATCH') return;
      } finally {
        if (mounted) setState(() => _busy = false);
      }
      if (!mounted) return;
    }
  }

  @override
  Widget build(BuildContext context) {
    final r = _result;
    final accept = r?['accept'] == true;
    final tone = accept ? kOk : kBad;
    return Scaffold(
      appBar: AppBar(title: const Text('Xác thực vân tay')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(children: [
              DropdownButtonFormField<String>(
                initialValue: _name,
                isExpanded: true,
                decoration: const InputDecoration(labelText: 'Người dùng'),
                items: [
                  for (final u in _users)
                    DropdownMenuItem(
                        value: u['name'] as String,
                        child: Text(u['name'] as String, overflow: TextOverflow.ellipsis)),
                ],
                onChanged: _busy
                    ? null
                    : (v) => setState(() {
                          _name = v;
                          _finger = _fingersOf.first;
                          _result = null;
                        }),
              ),
              const SizedBox(height: 20),
              DropdownButtonFormField<String>(
                initialValue: _finger,
                decoration: const InputDecoration(labelText: 'Ngón'),
                items: [
                  for (final f in _fingersOf)
                    DropdownMenuItem(value: f, child: Text('$f · ${fingers[f] ?? ''}')),
                ],
                onChanged: _busy
                    ? null
                    : (v) => setState(() {
                          _finger = v;
                          _result = null;
                        }),
              ),
            ]),
          ),
        ),
        const SizedBox(height: 24),
        FilledButton.icon(
          onPressed: _busy || _finger == null ? null : _verify,
          icon: const Icon(Icons.fingerprint),
          label: const Text('Quét và xác thực'),
        ),
        if (_busy)
          const Padding(
              padding: EdgeInsets.all(16), child: Center(child: CircularProgressIndicator())),
        if (_status != null)
          Padding(padding: const EdgeInsets.only(top: 16), child: StatusNote(_status!)),
        if (r != null)
          Container(
            key: ValueKey(r),
            margin: const EdgeInsets.only(top: 24),
            padding: const EdgeInsets.all(28),
            decoration: BoxDecoration(
              color: kCard,
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: tone, width: 1.5),
              boxShadow: [BoxShadow(color: tone.withValues(alpha: 0.3), blurRadius: 32)],
            ),
            child: Column(children: [
              TweenAnimationBuilder<double>(
                tween: Tween(begin: 0.5, end: 1),
                duration: const Duration(milliseconds: 350),
                curve: Curves.easeOutBack,
                builder: (_, t, child) => Transform.scale(scale: t, child: child),
                child: Icon(accept ? Icons.check_circle : Icons.cancel, size: 76, color: tone),
              ),
              const SizedBox(height: 20),
              Text(accept ? 'CHẤP NHẬN' : 'TỪ CHỐI',
                  style: Theme.of(context)
                      .textTheme
                      .headlineMedium
                      ?.copyWith(color: tone, fontWeight: FontWeight.w700)),
              const SizedBox(height: 6),
              Text(
                  'Điểm ${(r['score'] as num).toStringAsFixed(3)} · '
                  'ngưỡng ${(r['threshold'] as num).toStringAsFixed(3)}',
                  style: const TextStyle(color: kMuted)),
            ]),
          ),
      ]),
    );
  }
}
