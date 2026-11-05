// Danh sách người đã đăng ký; xóa khi người đó rút lại đồng ý lưu dữ liệu.
import 'package:flutter/material.dart';

import 'api.dart';

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
    final ok = await showDialog<bool>(context: context, builder: (ctx) => AlertDialog(
      title: Text('Xóa $name?'),
      content: const Text('Toàn bộ mẫu vân tay của người này sẽ bị xóa vĩnh viễn.'),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Hủy')),
        TextButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Xóa')),
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
                      ? ListView(children: const [
                          Padding(padding: EdgeInsets.all(32),
                              child: Center(child: Text('Chưa có ai đăng ký.'))),
                        ])
                      : ListView(children: [
                          for (final u in us)
                            ListTile(
                              leading: const Icon(Icons.person),
                              title: Text(u['name'] as String),
                              subtitle: Text((u['fingers'] as List)
                                  .map((f) => fingers[f] ?? f).join(', ')),
                              trailing: IconButton(
                                icon: const Icon(Icons.delete_outline),
                                tooltip: 'Xóa',
                                onPressed: () => _delete(u['name'] as String),
                              ),
                            ),
                        ]),
                ),
    );
  }
}
