import 'package:flutter/material.dart';

import 'api.dart';
import 'enroll_screen.dart';
import 'theme.dart';
import 'users_screen.dart';
import 'verify_screen.dart';

void main() => runApp(const FpApp());

class FpApp extends StatelessWidget {
  const FpApp({super.key});

  @override
  Widget build(BuildContext context) =>
      MaterialApp(title: 'FingerCam', theme: appTheme, home: const HomeScreen());
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  void _open(BuildContext context, Widget screen) =>
      Navigator.push(context, MaterialPageRoute(builder: (_) => screen));

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(actions: [
        IconButton(
          icon: const Icon(Icons.settings_outlined),
          tooltip: 'Cài đặt',
          onPressed: () => _open(context, const SettingsScreen()),
        ),
      ]),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: Column(children: [
            Expanded(
              child: Center(
                child: FittedBox(
                  fit: BoxFit.scaleDown,
                  child: Column(children: [
                    const _PulseLogo(),
                    const SizedBox(height: 24),
                    Text('FingerCam',
                        style: text.headlineLarge?.copyWith(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 6),
                    Text('Xác thực vân tay không chạm',
                        style: text.bodyLarge?.copyWith(color: kMuted)),
                  ]),
                ),
              ),
            ),
            _ActionCard(
              icon: Icons.person_add_alt_1,
              title: 'Đăng ký',
              subtitle: 'Thêm vân tay mới với $enrollSamples mẫu',
              onTap: () => _open(context, const EnrollScreen()),
            ),
            const SizedBox(height: 12),
            _ActionCard(
              icon: Icons.verified_user_outlined,
              title: 'Xác thực',
              subtitle: 'Kiểm tra vân tay đã đăng ký',
              onTap: () => _open(context, const VerifyScreen()),
            ),
            const SizedBox(height: 12),
            _ActionCard(
              icon: Icons.groups_outlined,
              title: 'Người dùng',
              subtitle: 'Xem và xóa dữ liệu đã lưu',
              onTap: () => _open(context, const UsersScreen()),
            ),
          ]),
        ),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String title, subtitle;
  final VoidCallback onTap;
  const _ActionCard(
      {required this.icon, required this.title, required this.subtitle, required this.onTap});

  @override
  Widget build(BuildContext context) => Card(
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: kAccent.withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: kAccent),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(title, style: Theme.of(context).textTheme.titleMedium),
                  Text(subtitle, style: const TextStyle(color: kMuted)),
                ]),
              ),
              const Icon(Icons.chevron_right, color: kMuted),
            ]),
          ),
        ),
      );
}

/// Vân tay trong vòng tròn phát sáng, có vòng lan tỏa chậm.
class _PulseLogo extends StatefulWidget {
  const _PulseLogo();

  @override
  State<_PulseLogo> createState() => _PulseLogoState();
}

class _PulseLogoState extends State<_PulseLogo> with SingleTickerProviderStateMixin {
  late final _anim = AnimationController(vsync: this, duration: const Duration(milliseconds: 2400));

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // tôn trọng tùy chọn giảm chuyển động của máy
    MediaQuery.disableAnimationsOf(context) ? _anim.stop() : _anim.repeat();
  }

  @override
  void dispose() {
    _anim.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => SizedBox(
        width: 200,
        height: 200,
        child: Stack(alignment: Alignment.center, children: [
          AnimatedBuilder(
            animation: _anim,
            builder: (_, __) => Container(
              width: 120 + 80 * _anim.value,
              height: 120 + 80 * _anim.value,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: kAccent.withValues(alpha: 0.5 * (1 - _anim.value))),
              ),
            ),
          ),
          Container(
            width: 120,
            height: 120,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: kAccent.withValues(alpha: 0.12),
              boxShadow: [BoxShadow(color: kAccent.withValues(alpha: 0.28), blurRadius: 40)],
            ),
            child: const Icon(Icons.fingerprint, size: 76, color: kAccent),
          ),
        ]),
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
          const SizedBox(height: 20),
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
