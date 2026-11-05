# App điện thoại (Flutter)

App chụp đầu ngón tay rồi gửi lên backend (`python -m server`) để **đăng ký** hoặc **xác thực**.
Toàn bộ xử lý vân tay chạy trên server; app chỉ chụp ảnh và hiện kết quả.

## Cài đặt lần đầu (Windows, Android)

1. Cài [Flutter SDK](https://docs.flutter.dev/get-started/install/windows/mobile), thêm `flutter\bin`
   vào `PATH`.
2. Cài [Android Studio](https://developer.android.com/studio). Mở *SDK Manager* và cài *Android SDK*
   cùng *Command-line Tools*. Chạy `flutter doctor --android-licenses`, rồi `flutter doctor` cho tới
   khi mục Android có dấu ✓.
3. Trên điện thoại Android: *Cài đặt → Thông tin điện thoại*, bấm 7 lần vào *Số hiệu bản tạo* để bật
   chế độ nhà phát triển. Sau đó bật *Gỡ lỗi USB* và cắm cáp vào máy tính.
4. Tạo các thư mục nền tảng. Lệnh này không ghi đè `lib/` và `pubspec.yaml` đã có:

   ```bash
   cd mobile
   flutter create . --platforms android,ios --project-name fpsys_app
   flutter pub get
   ```

5. Sửa `android/app/src/main/AndroidManifest.xml`:
   - Trong thẻ `<manifest>`, ngay trước `<application>`, thêm:

     ```xml
     <uses-permission android:name="android.permission.CAMERA"/>
     <uses-permission android:name="android.permission.INTERNET"/>
     ```

   - Thêm `android:usesCleartextTraffic="true"` vào thẻ `<application ...>`. Cần dòng này để gọi
     được server `http://` trong mạng LAN.
6. Plugin `camera` cần `minSdkVersion` 21 trở lên. Bản Flutter mới đã mặc định như vậy; nếu báo
   lỗi thì đặt `minSdk = 21` trong `android/app/build.gradle(.kts)`.
7. (iOS, cần máy Mac) Thêm `NSCameraUsageDescription` vào `ios/Runner/Info.plist`.

## Chạy

1. Trên máy tính, chạy server (xem README gốc, mục "Backend & app"), rồi lấy IP mạng LAN bằng
   `ipconfig`, ví dụ `192.168.1.5`.
2. Điện thoại và máy tính phải dùng **cùng Wi-Fi**. Nếu điện thoại không gọi được server, mở cổng
   8000 trên Windows Firewall.
3. `flutter run`. Trong app, vào ⚙ **Cài đặt**, nhập `http://192.168.1.5:8000` và API key
   (giá trị `FPSYS_API_KEY`).

## Cách dùng

- **Đăng ký:** nhập tên, chọn ngón, chụp 3 mẫu. Mỗi mẫu được server kiểm tra ngay; mẫu mờ, tối hoặc
  không thấy ngón sẽ bị báo lý do và phải chụp lại.
- **Xác thực:** chọn người và ngón, chụp 1 tấm. App hiện CHẤP NHẬN hoặc TỪ CHỐI, kèm điểm và ngưỡng.
- **Người dùng:** xem danh sách, xóa người không muốn lưu dữ liệu nữa.
- Khi chụp: đèn flash tự bật. Đặt đầu ngón vào khung oval, cách 8–15 cm, rồi **chạm vào đầu ngón trên
  màn hình** để lấy nét trước khi bấm Chụp.

## Cấu trúc

| File | Nội dung |
| --- | --- |
| `lib/api.dart` | Gọi 5 endpoint của server, lưu địa chỉ server và API key |
| `lib/capture_screen.dart` | Camera: độ phân giải tối đa, bật đèn, chạm để lấy nét, khung oval |
| `lib/enroll_screen.dart` | Luồng đăng ký 3 mẫu |
| `lib/verify_screen.dart` | Luồng xác thực |
| `lib/users_screen.dart` | Danh sách và xóa người dùng |
| `lib/main.dart` | Trang chủ và trang Cài đặt |
