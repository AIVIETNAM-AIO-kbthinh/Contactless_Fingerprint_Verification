# App điện thoại (Flutter)

App chụp đầu ngón tay rồi gửi lên backend (`python -m server`) để **đăng ký** hoặc **xác thực**.
Toàn bộ xử lý vân tay chạy trên server; app chỉ chụp ảnh và hiện kết quả. Đã thử trên Android
(Samsung SM-M556B, Android 16); bản iOS chưa được thử.

## Cài đặt lần đầu (Windows, Android)

1. Cài [Flutter SDK](https://docs.flutter.dev/get-started/install/windows/mobile) vào thư mục **ngoài
   repo**, ví dụ `D:\flutter` (không đặt trong thư mục dự án: SDK nặng ~1,4 GB và có `.git` riêng).
   Thêm `D:\flutter\bin` vào `PATH`, rồi **đóng hẳn VS Code/terminal và mở lại** để nhận `PATH` mới.
2. Cài [Android Studio](https://developer.android.com/studio). Trong *More Actions → SDK Manager →
   SDK Tools*, tích *Android SDK Command-line Tools (latest)* rồi *Apply*. Chạy
   `flutter doctor --android-licenses` (công cụ mới có thể báo "--licenses is no longer needed", không
   sao), rồi `flutter doctor` cho tới khi mục *Android toolchain* có dấu ✓. Mục *Visual Studio* báo
   thiếu thì bỏ qua (chỉ cần khi làm app Windows).
3. Trên điện thoại Android: *Cài đặt → Thông tin điện thoại → Thông tin phần mềm*, bấm 7 lần vào
   *Số hiệu bản tạo* để bật chế độ nhà phát triển, rồi bật *Gỡ lỗi USB*. Cắm cáp, bấm **Cho phép**
   trên điện thoại. `flutter devices` phải hiện tên máy.
4. Tải các package của app:

   ```powershell
   cd mobile
   flutter pub get
   ```

Thư mục `android/` đã có sẵn trong repo, kèm quyền camera, quyền mạng và cho phép gọi `http://`
trong mạng LAN (`android/app/src/main/AndroidManifest.xml`).

(iOS, cần máy Mac) Thêm `NSCameraUsageDescription` và ngoại lệ App Transport Security cho `http://`
vào `ios/Runner/Info.plist`.

## Chạy

1. Chạy server trên máy tính (README gốc, mục "Backend & app").
2. Chạy app: `flutter run` (trong thư mục `mobile`). Lần build đầu mất vài phút.
3. Kết nối app với server, chọn một cách:
   - **Qua cáp USB (ổn định nhất):** chạy
     `& "$env:LOCALAPPDATA\Android\sdk\platform-tools\adb.exe" reverse tcp:8000 tcp:8000`, trong app
     nhập địa chỉ `http://localhost:8000`. Chạy lại lệnh mỗi lần cắm lại cáp.
   - **Qua Wi-Fi:** điện thoại và máy tính cùng một Wi-Fi; lấy IP Wi-Fi của máy tính bằng `ipconfig`,
     nhập `http://<IP>:8000`. Mở cổng 8000 (PowerShell **Administrator**), và đặt Wi-Fi là mạng Private
     nếu đang Public:

     ```powershell
     New-NetFirewallRule -DisplayName "fpsys server" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
     Set-NetConnectionProfile -InterfaceAlias "Wi-Fi" -NetworkCategory Private
     ```

4. Trong app, vào ⚙ **Cài đặt**, nhập địa chỉ server và API key (giá trị `FPSYS_API_KEY`). Mở
   **Người dùng**: hiện "Chưa có ai đăng ký" là đã kết nối.

## Build APK để phát hành (người dùng chỉ cần cài APK)

Sau khi triển khai server lên VPS (README gốc, mục "Triển khai lên VPS"), chạy lệnh mà
`deploy/setup.sh` in ra. Lệnh có dạng:

```powershell
cd mobile
flutter build apk --release --split-per-abi --dart-define=SERVER_URL=https://<IP>.sslip.io --dart-define=API_KEY=<key>
```

File cần gửi: `build/app/outputs/flutter-apk/app-arm64-v8a-release.apk`. Người dùng cài là dùng được
ngay, không cần nhập địa chỉ hay API key, không cần cáp hay chung Wi-Fi.

- Khi cài, Android sẽ hỏi cho phép **cài ứng dụng không rõ nguồn gốc**; Play Protect có thể cảnh báo,
  chọn "Vẫn cài đặt". Nếu đang có bản cài bằng `flutter run`, gỡ bản đó trước.
- APK được ký bằng khóa debug của máy build. Muốn cập nhật app mà không phải gỡ, hãy build bản mới
  trên **cùng máy**.
- **Ai có file APK đều lấy được API key**, nên chỉ gửi APK cho người tin cậy. Nếu lộ, đổi
  `FPSYS_API_KEY` trong `/etc/fpsys.env` trên VPS, `sudo systemctl restart fpsys`, rồi build lại APK.
- Không có `--dart-define` (ví dụ khi `flutter run`), app vẫn mặc định `http://192.168.1.5:8000` như
  trước. Giá trị nhập trong ⚙ **Cài đặt** luôn được ưu tiên, nên vẫn đổi qua lại giữa VPS và máy tính
  được.

## Cách dùng

- **Đăng ký:** nhập tên, chọn ngón, chụp 3 mẫu. Mỗi mẫu được server kiểm tra ngay; mẫu mờ, tối hoặc
  không thấy ngón sẽ bị báo lý do và phải chụp lại.
- **Xác thực:** chọn người và ngón, chụp 1 tấm. App hiện CHẤP NHẬN hoặc TỪ CHỐI, kèm điểm và ngưỡng.
- **Người dùng:** xem danh sách, xóa người không muốn lưu dữ liệu nữa.
- Khi chụp: đèn tự bật, camera tự zoom 2x. Đặt đầu ngón vào khung oval, cách **10–15 cm** (camera chính
  không lấy nét được gần hơn), mặt bụng ngón nhìn thẳng camera, **chạm vào giữa đầu ngón** để lấy nét,
  chờ 1 giây rồi bấm Chụp. Giữ cùng tư thế cho cả 3 mẫu đăng ký.

## Xử lý sự cố

| Hiện tượng | Cách xử lý |
| --- | --- |
| `flutter` không nhận | `PATH` chưa có `...\flutter\bin`, hoặc chưa mở lại VS Code/terminal |
| `flutter devices` không thấy điện thoại | Đổi cáp (cáp chỉ sạc không truyền dữ liệu), chọn chế độ USB *Truyền tệp*, bật *Gỡ lỗi USB*, bấm *Cho phép* |
| Build lỗi `did not install NDK 28.2.13676358` | `& "$env:LOCALAPPDATA\Android\sdk\cmdline-tools\latest\bin\android.exe" sdk install ndk/28.2.13676358` rồi build lại |
| App báo "Không kết nối được server" | Server chưa chạy, khác Wi-Fi, tường lửa chặn, hoặc dùng cách qua cáp USB ở trên |
| Đăng ký luôn báo các mẫu không khớp | Chạy server với `--set enroll.min_consistency=0.08` (README gốc) |

## Cấu trúc

| File | Nội dung |
| --- | --- |
| `lib/api.dart` | Gọi 5 endpoint của server, lưu địa chỉ server và API key |
| `lib/capture_screen.dart` | Camera: độ phân giải tối đa, bật đèn, zoom 2x, chạm để lấy nét, khung oval |
| `lib/enroll_screen.dart` | Luồng đăng ký 3 mẫu |
| `lib/verify_screen.dart` | Luồng xác thực |
| `lib/users_screen.dart` | Danh sách và xóa người dùng |
| `lib/main.dart` | Trang chủ và trang Cài đặt |
| `android/gradle.properties` | `kotlin.incremental=false`: tránh lỗi Kotlin khi dự án và cache package ở hai ổ đĩa khác nhau |
