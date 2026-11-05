// Xác thực 1:1: chọn người + ngón đã đăng ký, chụp 1 tấm, hiện CHẤP NHẬN / TỪ CHỐI.
import 'package:flutter/material.dart';

import 'api.dart';
import 'capture_screen.dart';

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
      final us = (await (await Api.load()).users()).cast<Map>()
          .where((u) => (u['fingers'] as List).isNotEmpty).toList();
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

  Future<void> _verify() async {
    final path = await Navigator.push<String>(
        context, MaterialPageRoute(builder: (_) => const CaptureScreen(title: 'Xác thực')));
    if (path == null) return;
    setState(() {
      _busy = true;
      _result = null;
      _status = 'Đang xác thực…';
    });
    try {
      final r = await (await Api.load()).verify(_name!, _finger!, path);
      setState(() {
        _result = r;
        _status = null;
      });
    } on ApiError catch (e) {
      setState(() => _status = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final r = _result;
    final accept = r?['accept'] == true;
    return Scaffold(
      appBar: AppBar(title: const Text('Xác thực vân tay')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        DropdownButtonFormField<String>(
          value: _name,
          decoration: const InputDecoration(labelText: 'Người dùng'),
          items: [
            for (final u in _users)
              DropdownMenuItem(value: u['name'] as String, child: Text(u['name'] as String)),
          ],
          onChanged: _busy ? null : (v) => setState(() {
            _name = v;
            _finger = _fingersOf.first;
            _result = null;
          }),
        ),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
          value: _finger,
          decoration: const InputDecoration(labelText: 'Ngón'),
          items: [
            for (final f in _fingersOf)
              DropdownMenuItem(value: f, child: Text('$f · ${fingers[f] ?? ''}')),
          ],
          onChanged: _busy ? null : (v) => setState(() {
            _finger = v;
            _result = null;
          }),
        ),
        const SizedBox(height: 24),
        FilledButton.icon(
          onPressed: _busy || _finger == null ? null : _verify,
          icon: const Icon(Icons.fingerprint),
          label: const Text('Chụp và xác thực'),
        ),
        if (_busy) const Padding(padding: EdgeInsets.all(16),
            child: Center(child: CircularProgressIndicator())),
        if (_status != null) Padding(padding: const EdgeInsets.only(top: 16), child: Text(_status!)),
        if (r != null)
          Card(
            margin: const EdgeInsets.only(top: 24),
            color: accept ? Colors.green.shade100 : Colors.red.shade100,
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(children: [
                Icon(accept ? Icons.check_circle : Icons.cancel, size: 64,
                    color: accept ? Colors.green.shade800 : Colors.red.shade800),
                const SizedBox(height: 8),
                Text(accept ? 'CHẤP NHẬN' : 'TỪ CHỐI',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        color: accept ? Colors.green.shade900 : Colors.red.shade900)),
                Text('Điểm ${(r['score'] as num).toStringAsFixed(3)} · '
                    'ngưỡng ${(r['threshold'] as num).toStringAsFixed(3)}',
                    style: const TextStyle(color: Colors.black87)),
              ]),
            ),
          ),
      ]),
    );
  }
}
