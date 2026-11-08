// Giao diện chung: bảng màu tối kiểu sinh trắc học và các thành phần dùng ở nhiều màn.
import 'package:flutter/material.dart';

const kBg = Color(0xFF0B1416);
const kCard = Color(0xFF142225);
const kField = Color(0xFF1C2F33);
const kAccent = Color(0xFF2DD4BF);
const kText = Color(0xFFE6F1F0);
const kMuted = Color(0xFF8FA5A6);
const kOk = Color(0xFF34D399);
const kBad = Color(0xFFF87171);

final appTheme = ThemeData(
  useMaterial3: true,
  brightness: Brightness.dark,
  colorScheme: const ColorScheme.dark(
    primary: kAccent,
    onPrimary: Color(0xFF04201C),
    secondary: kAccent,
    onSecondary: Color(0xFF04201C),
    surface: kCard,
    onSurface: kText,
    error: kBad,
  ),
  scaffoldBackgroundColor: kBg,
  canvasColor: kCard, // nền danh sách thả xuống
  appBarTheme: const AppBarTheme(
    backgroundColor: Colors.transparent,
    surfaceTintColor: Colors.transparent,
    scrolledUnderElevation: 0,
    foregroundColor: kText,
  ),
  cardTheme: CardThemeData(
    color: kCard,
    elevation: 0,
    margin: EdgeInsets.zero,
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
  ),
  inputDecorationTheme: InputDecorationTheme(
    filled: true,
    fillColor: kField,
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(14),
      borderSide: BorderSide.none,
    ),
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(14),
      borderSide: const BorderSide(color: kAccent, width: 1.5),
    ),
  ),
  filledButtonTheme: FilledButtonThemeData(
    style: FilledButton.styleFrom(
      minimumSize: const Size.fromHeight(52),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
    ),
  ),
);

/// Thông báo trạng thái dạng thẻ có icon.
class StatusNote extends StatelessWidget {
  final String text;
  const StatusNote(this.text, {super.key});

  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Icon(Icons.info_outline, color: kAccent, size: 20),
            const SizedBox(width: 10),
            Expanded(child: Text(text)),
          ]),
        ),
      );
}
