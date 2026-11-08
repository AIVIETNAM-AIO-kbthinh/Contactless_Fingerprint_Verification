# Contactless Fingerprint Verification — `fpsys`

Lõi xác thực vân tay **1:1 không tiếp xúc**: ảnh chụp đầu ngón bằng camera điện thoại → phát hiện
đầu ngón (YOLO11n) → chuẩn hóa (mask da, xoay, duỗi phẳng, chuẩn hóa tỷ lệ về ~500 ppi) → kiểm tra
chất lượng → tăng cường Gabor → làm mảnh → minutiae (Crossing Number) → đối sánh MCC (Minutia
Cylinder-Code + LSS-R). Kế hoạch chi tiết: [fpsys_plan.md](fpsys_plan.md).

## Cài đặt

```bash
# Python ≥ 3.10, chỉ cần CPU
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
# (tùy chọn, cho ablation ref_enhancer) uv pip install fingerprint-enhancer
```

Weights YOLO: `yolo11n_640/weights/best.pt` có sẵn trong repo; trên máy mới cần chép sang
`models/best.pt` (thư mục `models/`, `data/`, `results/` được gitignore):

```powershell
New-Item -ItemType Directory -Force models; Copy-Item yolo11n_640\weights\best.pt models\best.pt
```

> Không cài `opencv-contrib-python` (xung đột `cv2` với bản `ultralytics` kéo theo). Code không
> dùng `cv2.ximgproc`.

## API cho backend

```python
import fpsys

cfg = fpsys.load_config()                      # configs/default.yaml (+ overrides=["a.b=c"])
img = fpsys.load_image(path_or_bytes)          # BGR uint8, đã áp xoay EXIF
t = fpsys.extract(img, cfg)                    # Template; lỗi chụp → fpsys.CaptureError(code)
refs = fpsys.select_enrollment([t1, t2, t3], cfg)   # loại mẫu enroll lạc
res = fpsys.verify(probe, refs, cfg)           # VerifyResult(score, threshold, accept, per_template)
s = fpsys.match(a, b, cfg)                     # điểm [0, 1], đối xứng
d = t.to_dict(); t2 = fpsys.Template.from_dict(d)   # lưu trữ JSON
```

Mã lỗi `CaptureError.code`: `NO_FINGER`, `BLURRY`, `BAD_EXPOSURE`, `LOW_QUALITY`,
`TOO_FEW_MINUTIAE`, `INCONSISTENT_ENROLL`. `match()` raise `TemplateMismatchError` khi hai template
có `meta["cfg_hash"]` khác nhau (đổi các mục `pipeline/normalize/enhance/skeleton/minutiae` của
config ⇒ phải enroll lại).

Điều kiện chụp (cần đưa vào hướng dẫn trên app): một ngón, bật flash, khoảng cách 8–15 cm, chạm để
lấy nét vào đầu ngón (hoặc zoom 2x), ảnh JPEG độ phân giải đầy đủ.

## CLI

```bash
python -m fpsys extract IMG [--debug-dir out/] [--set enhance.invert=false]
python -m fpsys enroll --user U01 --finger R2 IMG1 IMG2 IMG3
python -m fpsys verify --user U01 --finger R2 IMG
python -m fpsys match IMG_A IMG_B
python -m fpsys viz IMG --out out.png          # lưới 3×3 ảnh 9 bước
```

Mã thoát: 0 chấp nhận/thành công, 1 từ chối, 2 `CaptureError`. Ảnh tiếp xúc (FVC):
`--set pipeline.source=contact`; ảnh đã crop sẵn: `--set pipeline.source=contactless_crop`.
Template enroll được lưu ở `data/store.json` (chỉ để thử nghiệm).

## Giao diện thử nghiệm (`python -m demo`)

App desktop (Tkinter, không cần cài thêm): đăng ký tên, chụp vân tay bằng camera có hướng dẫn trực
tiếp, xác thực.

```bash
python -m demo                                        # webcam mặc định
python -m demo --camera 1                             # webcam khác
python -m demo --camera http://192.168.1.5:8080/video # điện thoại làm camera (app IP Webcam)
```

- **Hướng dẫn trực tiếp** (mỗi khung hình ~30 ms): không thấy ngón, nhiều ngón, lệch khung, quá
  gần/xa, thiếu sáng/quá sáng/cháy sáng, rung tay, chưa thấy rõ vân. Khi đạt liên tiếp
  `guide.ready_frames` lần thì báo "Sẵn sàng" và tự chụp nếu bật. Ngưỡng dùng chung với pipeline;
  ngưỡng riêng ở mục `guide` của config.
- **Đăng ký:** nhập tên, chọn ngón, chụp `guide.enroll_samples` mẫu → `select_enrollment` → lưu.
  **Xác thực:** chọn người + ngón → chụp → điểm, ngưỡng, CHẤP NHẬN/TỪ CHỐI.
- Danh sách người dùng và template của demo lưu ở `data/demo/` (tên → mã `S01`, `S02`…).
- **Thu dữ liệu:** bật "Lưu ảnh vào data/own" để mỗi lần chụp được lưu đúng quy ước
  `S01_R2_s1_c_01.jpg` (chọn phiên/điều kiện), dùng thẳng cho `eval`. Chỉ lưu khi người được chụp
  đồng ý.
- "Mở ảnh từ file…" để thử với ảnh chụp sẵn; "Xem 9 bước" để xem ảnh trung gian của lần chụp gần
  nhất.

> Webcam laptop (thường 720p, lấy nét cố định) gần như không đủ để thấy vân tay. Nên dùng điện thoại
> làm camera qua Wi‑Fi (Android: app *IP Webcam*, nhập `http://<ip>:8080/video`), hoặc chụp bằng
> điện thoại rồi mở file gốc.

## Backend & app (`python -m server`, `mobile/`)

Backend FastAPI bọc API của `fpsys`. Template lưu trong SQLite (`data/server.db`), mã hóa bằng
Fernet. Mọi request cần header `X-API-Key`.

```bash
.venv/Scripts/python -m pip install -e ".[server]"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# PowerShell: $env:FPSYS_DB_KEY="<khóa vừa tạo>"; $env:FPSYS_API_KEY="<mật khẩu tùy chọn>"
python -m server [--port 8000] [--set verify.threshold=0.12]    # tài liệu API: http://localhost:8000/docs
```

Giữ nguyên `FPSYS_DB_KEY` giữa các lần chạy: đổi khóa thì không giải mã được template cũ.

Trên ảnh thật chụp bằng điện thoại, minutiae (MCC) cho điểm rất thấp: cùng ngón có trung vị 0,027, gần
mức nhiễu. `deploy/setup.sh` vì vậy bật so khớp SIFT: `--set sift.enabled=true --set matcher.method=sift
--set enroll.consistency=max --set enroll.min_consistency=0.30 --set verify.threshold=0.30`. Các ngưỡng
này **chưa hiệu chỉnh trên tập dev**: chúng được chọn từ 8 ảnh cùng buổi, gấp khoảng hai lần điểm cao
nhất của ảnh lật gương (0,167). Khi demo hãy thử cả ngón khác để chắc chắn bị TỪ CHỐI. Chạy server trên
máy tính mà không có các cờ này thì vẫn dùng MCC; hai cách dùng template khác nhau nên phải đăng ký lại
khi đổi.

| Method | Path | Kết quả |
| --- | --- | --- |
| POST | `/check` (`image`) | `{ok, n_minutiae}` hoặc `{ok: false, code, message}` |
| POST | `/users/{name}/fingers/{finger}/enroll` (`images` ×N) | `{kept, total}` |
| POST | `/users/{name}/fingers/{finger}/verify` (`image`) | `{score, threshold, accept}` |
| GET | `/users` | `[{name, fingers}]` |
| DELETE | `/users/{name}` | 204 |

`CaptureError` → 422 `{code, message}` (câu hướng dẫn tiếng Việt lấy từ `demo/session.py::HINTS`).
Chưa đăng ký → 404; template khác `cfg_hash` → 409. App Flutter: xem [mobile/README.md](mobile/README.md).

### Triển khai lên VPS (chạy 24/7, HTTPS)

Cần VPS Ubuntu 22.04/24.04, tối thiểu 1 vCPU / 1 GB RAM / 20 GB disk, mở cổng 22, 80, 443 (kiểm tra cả
tường lửa trên trang quản lý của nhà cung cấp VPS). Không cần tên miền: script dùng `<IP>.sslip.io` và
Caddy tự xin chứng chỉ HTTPS.

```bash
sudo git clone https://github.com/AIVIETNAM-AIO-kbthinh/Contactless_Fingerprint_Verification.git /opt/fpsys
cd /opt/fpsys && sudo bash deploy/setup.sh
```

Nếu repo private, GitHub sẽ hỏi đăng nhập: nhập username và một Personal Access Token thay cho mật
khẩu. Lần đầu script mất 10–20 phút (tải torch). Cuối cùng script in ra địa chỉ `https://...sslip.io`
và lệnh build APK có sẵn API key.

- **Dịch vụ:** `fpsys` (server, chỉ nghe `127.0.0.1:8000`) và `caddy` (HTTPS) tự khởi động lại khi
  lỗi hoặc khi VPS reboot.
- **Xem log:** `journalctl -u fpsys -f`. **Khởi động lại:** `sudo systemctl restart fpsys`.
- **Cập nhật code:** `cd /opt/fpsys && sudo git pull && sudo bash deploy/setup.sh` (giữ nguyên khóa và
  dữ liệu).
- **Khóa và dữ liệu:** `/etc/fpsys.env` (`FPSYS_DB_KEY`, `FPSYS_API_KEY`) và `/var/lib/fpsys/server.db`.
  Hãy sao lưu cả hai: mất `FPSYS_DB_KEY` là mất toàn bộ dữ liệu đã đăng ký.
- Server xử lý **từng ảnh một** để VPS 1 GB không hết RAM; nhiều người chụp cùng lúc thì phải chờ nhau
  (~3–6 giây mỗi ảnh trên 1 vCPU).

## Đánh giá

Dữ liệu tự thu đặt ở `data/own/`, tên file `{subject}_{finger}_{session}_{cond}_{sample}.jpg`
(ví dụ `S03_R2_s1_c_04.jpg`). Enroll = `s1_c`, probe = `s2_c` (có kiểm soát) hoặc `s2_u`.

```bash
python -m eval.splits --dataset own                          # chia subject dev/test cố định
python -m eval.run_eval --dataset own --split dev --cond c   # → results/<run>/
python -m eval.run_eval --dataset own --split test --cond c --dev-run <run_dev>
python -m eval.run_eval --dataset own --split dev --cond c --ablation   # → results/ablation.csv
python -m eval.run_eval --dataset own --split dev --timing             # 50 ảnh, mean/p95 từng bước
python -m eval.run_eval --dataset fvc                        # FVC2002 DB1_B ở data/fvc2002_db1b/
```

Mỗi lần chạy ghi `scores.csv` (mức template), `scores_verify.csv` (mức hệ thống: probe so với tập
enroll đã qua `select_enrollment`, gộp theo `verify.fusion`), `metrics.json` (EER, CI 95% bootstrap
theo subject, FRR@FAR=1%, FTA theo mã lỗi, FRR tổng quát, EER impostor cùng người,
`threshold_eer`, `threshold_at_far`), `det.png`, `hist.png`, `fta.csv`, `config.yaml`.
Template được cache ở `data/cache/`. Thêm `--workers N` để trích song song.

**Quy tắc:** mọi tinh chỉnh (ngưỡng quality, ablation, `verify.threshold`) chỉ dùng tập dev. Ngưỡng
khuyến nghị cho `verify.threshold` là `recommended_threshold` (= `threshold_at_far` của tập dev).

RidgeBase: parser chưa viết — cần xem cấu trúc thư mục thực tế và thống nhất giao thức trước.

Baseline ngoài (tùy chọn): `python -m eval.baselines.export_enhanced ...` xuất ảnh tăng cường +
`pairs.csv`; chạy `eval/baselines/SourceAfisScores.java` (cần JDK + jar SourceAFIS), rồi
`python -m eval.baselines.score_csv scores.csv`.

## Phát triển

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/ruff check .
```

Test dùng ảnh tổng hợp sinh bằng code (`tests/synth.py`) và YOLO giả, không cần dữ liệu thật.

## Ghi chú triển khai (khác / thêm so với plan)

Các khóa config thêm vào `configs/default.yaml` (đều có giá trị trả về hành vi gốc):

| Khóa | Mặc định | Lý do |
| --- | --- | --- |
| `sift.*`, `matcher.method=sift` | tắt | So khớp SIFT trên ảnh đã làm phẳng sáng (`fpsys/match_sift.py`). Trên 8 ảnh thật: AUC 0.99 so với 0.91 của MCC. Mặc định tắt nên eval không đổi. |
| `enroll.consistency` | `median` | `max`: giữ mẫu nếu khớp ít nhất một mẫu khác, để các mẫu đăng ký được khác góc nhau. `median` = như plan. |
| `matcher.mcc.relax.scale_tol` | 0.06 | LSS-R gốc phạt chênh lệch khoảng cách tuyệt đối (`mu1 = 5 px`), nên sai số tỷ lệ 5% giữa hai lần chụp làm điểm tụt từ ~0.9 xuống 0.10. Trừ đi dung sai tương đối trước khi áp sigmoid. `0` = như bài báo. |
| `matcher.mcc.relax.support` | `weighted` | Hỗ trợ trong relaxation = trung bình ρ·λ có trọng số λ, thay vì chia cho `n_R − 1`. Cặp rác (λ nhỏ) không còn làm loãng cặp thật: test 20% điểm thêm/bớt tăng từ 0.41 lên 0.61, impostor ngẫu nhiên vẫn < 0.02. `mean` = như bài báo. |
| `minutiae.shift_frac` | 0.5 | Dời ending tiến ½ chu kỳ, bifurcation lùi ½ chu kỳ dọc θ, để ending trên vân và bifurcation đối ngẫu trên rãnh trùng vị trí (≤ 3 px). Nếu không dời, chúng lệch nhau khoảng một chu kỳ vân. |
| `minutiae.break_angle_deg`, `cluster_max` | 45, 4 | Tham số hóa luật lọc "hướng gần đối nhau" và "nhiều hơn 4 điểm". |
| `matcher.mcc.min_common_cells` | 0.6 | Tỷ lệ ô chung tối thiểu để so hai cylinder (minME trong MCC). |
| `normalize.scale_norm` | true | Ablation `no_scale_norm`. |
| `normalize.unwarp_smooth_rows` | 31 | Cửa sổ median filter cho c(y), r(y) khi duỗi phẳng. |
| `normalize.contact_block`, `contact_std_frac` | 16, 0.25 | Mask ảnh tiếp xúc từ độ lệch chuẩn khối. |
| `enhance.freq_block` | 32 | Khối ước lượng tần số vân. |
| `enhance.method` | `hong` | Ablation `ref_enhancer` (`reference`). |
| `detect.method` | `yolo` | Ablation `otsu_crop` (`otsu`). Khóa cache eval gồm cả mục `detect`. |
| `enhance.gabor_gamma` | 1.0 | Tỷ lệ σ vuông góc / σ dọc vân của kernel Gabor; < 1 kéo dài kernel theo vân. Thử 0.5 trên dev: không cải thiện. |
| `enhance.channel` | `gray` | Thêm lựa chọn `blue` (ngoài `gray`, `green`). |

Một số điểm khác:
- `Detection` có thêm trường `crop_xyxy` (vùng crop sau pad) để `grabCut` khởi tạo bằng box YOLO trước khi pad.
- `pad` nới **mỗi phía** thêm `pad × kích thước box`.
- Khi không ước lượng được chu kỳ vân, `normalize` trả `BAD_EXPOSURE` nếu phơi sáng đã sai ngưỡng, ngược lại trả `BLURRY`.
- `run_eval` báo cả hai mức: template (cặp ảnh–ảnh như mục 8.2) và verify (probe so với tập enroll, gộp theo `verify.fusion`; ablation `mean_top2` cần mức này). Ngưỡng khuyến nghị lấy theo mức verify.
- Thời gian đo trên ảnh tổng hợp 4000×3000 (crop 1000×1400): YOLO + TTA ~0.17 s, pipeline ~1–1.6 s. MCC 60×60 điểm: ~3 ms khi cylinder đã cache, ~45 ms nếu tính cả dựng cylinder.

### Hiệu chỉnh trên ảnh thật (2026-11-03, bộ `data/own/Vân tay test`)

Ba người, nhãn ngón do Claude gán theo giờ chụp + so ảnh (`labels.csv`, cột `do_tin_cay`). Dev = Tống An (chọn tham số), test = Thịnh + Đức (chỉ đánh giá). Giao thức: mọi cặp ảnh hợp lệ.

| Cấu hình | EER dev | EER test |
| --- | --- | --- |
| Cũ (`unwarp: true`, `orient_sigma: 7`) | 39.7% | 36.3% |
| Mới (`unwarp: false`, `orient_sigma: 14`) | 25.7% | 25.7% |

- Nguyên nhân gốc: biên độ vân trong ảnh chỉ ~2 mức xám (std dải tần vân), ngang nhiễu tần số cao (SNR ~1–1.6). Thêm nhiễu σ = 2 vào chính một ảnh đã làm minutiae xáo trộn.
- `unwarp` khuếch đại dao động của mask GrabCut (nhiễu nhỏ làm góc xoay chuẩn hóa đổi tới 5°); MCC vốn bất biến xoay nên tắt.
- Đã thử, không giúp trên dev: kênh `green`, `target_period: 12`, `min_coherence` 0.1/0.2, bỏ CLAHE, làm mượt trước khi tính hướng, Gabor dài (`gabor_gamma: 0.5`), nới ngưỡng relaxation MCC, Hough.
- Còn tồn tại: luật xác định đầu ngón sai ở 11/69 ảnh (xoay ngược 180°, không ảnh hưởng MCC nhưng ảnh hưởng Hough `max_rotation_deg`); FTA 40% (chủ yếu ảnh quá sáng).

## Trạng thái milestone

| Milestone | Nội dung | Trạng thái |
| --- | --- | --- |
| M0 | Khung dự án, types, config, io | xong |
| M1 | detect (YOLO + TTA xoay) | xong (test mock); **chờ kiểm tra crop trên 20 ảnh thật** |
| M2 | normalize, quality | xong (test tổng hợp); **ngưỡng quality cần hiệu chỉnh trên ảnh thật** |
| M3 | enhance | xong (test tổng hợp); **chờ duyệt bằng mắt `04_flat`/`05_enhanced`, chốt `invert`/`channel`** |
| M4 | skeleton, minutiae | xong (test tổng hợp); chờ FVC để kiểm tra 20–80 minutiae/ảnh |
| M5 | MCC, Hough, matcher | xong (test tổng hợp); chờ FVC để đo EER |
| M6 | pipeline, decision, store, CLI | xong; chờ ảnh tự thu để đo thời gian / FTA |
| M7 | đánh giá | code xong; chờ dữ liệu |
| M8 | embedding + fusion | chưa làm (tùy chọn) |
