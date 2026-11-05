import 'package:flutter/material.dart';

import 'api.dart';
import 'enroll_screen.dart';
import 'users_screen.dart';
import 'verify_screen.dart';

void main() => runApp(const FpApp());

class FpApp extends StatelessWidget {
  const FpApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Vân tay không chạm',
        theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
        home: const HomeScreen(),
      );
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  void _open(BuildContext context, Widget screen) =>
      Navigator.push(context, MaterialPageRoute(builder: (_) => screen));

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Vân tay không chạm'), actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: 'Cài đặt',
            onPressed: () => _open(context, const SettingsScreen()),
          ),
        ]),
        body: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            const Icon(Icons.fingerprint, size: 96),
            const SizedBox(height: 32),
            FilledButton.icon(
              onPressed: () => _open(context, const EnrollScreen()),
              icon: const Icon(Icons.person_add),
              label: const Text('Đăng ký'),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: () => _open(context, const VerifyScreen()),
              icon: const Icon(Icons.verified_user),
              label: const Text('Xác thực'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: () => _open(context, const UsersScreen()),
              icon: const Icon(Icons.people),
              label: const Text('Người dùng'),
            ),
          ]),
        ),
      );
}

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _url = TextEditingController();
  final _key = TextEditingController();

  @override
  void initState() {
    super.initState();
    Api.load().then((a) => setState(() {
          _url.text = a.baseUrl;
          _key.text = a.apiKey;
        }));
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Cài đặt')),
        body: ListView(padding: const EdgeInsets.all(16), children: [
          TextField(
            controller: _url,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
                labelText: 'Địa chỉ server', hintText: 'http://192.168.1.5:8000'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _key,
            obscureText: true,
            decoration: const InputDecoration(labelText: 'API key (FPSYS_API_KEY)'),
          ),
          const SizedBox(height: 24),
          FilledButton(
            onPressed: () async {
              await Api.save(_url.text, _key.text);
              if (context.mounted) Navigator.pop(context);
            },
            child: const Text('Lưu'),
          ),
        ]),
      );
}
