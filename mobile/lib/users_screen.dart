// Danh sách người đã đăng ký; xóa khi người đó rút lại đồng ý lưu dữ liệu.
import 'package:flutter/material.dart';

import 'api.dart';
import 'theme.dart';

class UsersScreen extends StatefulWidget {
  const UsersScreen({super.key});

  @override
  State<UsersScreen> createState() => _UsersScreenState();
}

class _UsersScreenState extends State<UsersScreen> {
  List<Map>? _users;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final us = (await (await Api.load()).users()).cast<Map>();
      setState(() {
        _users = us;
        _error = null;
      });
    } on ApiError catch (e) {
      setState(() => _error = e.message);
    }
  }

  Future<void> _delete(String name) async {
    final ok = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
              title: Text('Xóa $name?'),
              content: const Text('Toàn bộ mẫu vân tay của người này sẽ bị xóa vĩnh viễn.'),
              actions: [
                TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Hủy')),
                TextButton(
                  onPressed: () => Navigator.pop(ctx, true),
                  child: const Text('Xóa', style: TextStyle(color: kBad)),
                ),
              ],
            ));
    if (ok != true) return;
    try {
      await (await Api.load()).deleteUser(name);
    } on ApiError catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      }
    }
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    final us = _users;
    return Scaffold(
      appBar: AppBar(title: const Text('Người dùng')),
      body: _error != null
          ? Center(child: Padding(padding: const EdgeInsets.all(16), child: Text(_error!)))
          : us == null
              ? const Center(child: CircularProgressIndicator())
              : RefreshIndicator(
                  onRefresh: _load,
                  child: us.isEmpty
                      ? ListView(padding: const EdgeInsets.all(48), children: const [
                          Icon(Icons.fingerprint, size: 72, color: kMuted),
                          SizedBox(height: 12),
                          Text('Chưa có ai đăng ký',
                              textAlign: TextAlign.center, style: TextStyle(fontSize: 18)),
                          SizedBox(height: 4),
                          Text('Về trang chủ và chọn Đăng ký để thêm vân tay đầu tiên.',
                              textAlign: TextAlign.center, style: TextStyle(color: kMuted)),
                        ])
                      : ListView.separated(
                          padding: const EdgeInsets.all(16),
                          itemCount: us.length,
                          separatorBuilder: (_, __) => const SizedBox(height: 12),
                          itemBuilder: (_, i) => _userCard(us[i]),
                        ),
                ),
    );
  }

  Widget _userCard(Map u) {
    final name = u['name'] as String;
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 12, 6, 12),
        child: Row(children: [
          CircleAvatar(
            radius: 24,
            backgroundColor: kAccent.withValues(alpha: 0.16),
            child: Text(name.isEmpty ? '?' : name.characters.first.toUpperCase(),
                style: const TextStyle(color: kAccent, fontSize: 20, fontWeight: FontWeight.w700)),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 6),
              Wrap(spacing: 6, runSpacing: 6, children: [
                for (final f in u['fingers'] as List)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration:
                        BoxDecoration(color: kField, borderRadius: BorderRadius.circular(10)),
                    child: Text('${fingers[f] ?? f}', style: const TextStyle(fontSize: 12)),
                  ),
              ]),
            ]),
          ),
          IconButton(
            icon: const Icon(Icons.delete_outline, color: kBad),
            tooltip: 'Xóa',
            onPressed: () => _delete(name),
          ),
        ]),
      ),
    );
  }
}
