# fpsys — Kế hoạch triển khai lõi hệ thống xác thực vân tay không tiếp xúc

> Tài liệu này dành cho coding agent. Phạm vi: **lõi xử lý vân tay (`fpsys`) và workflow enroll/verify/đánh giá**. Backend API, ứng dụng di động và phần bảo mật do thành viên khác phụ trách; ở đây chỉ định nghĩa giao diện mà họ sẽ gọi.
> Làm theo thứ tự các milestone ở mục 9. Mỗi milestone có tiêu chí hoàn thành; chỉ chuyển sang milestone tiếp theo khi tiêu chí trước đã đạt. M8 là tùy chọn, chỉ làm khi người dùng yêu cầu.

---

## 1. Giới thiệu bài toán

### 1.1. Bối cảnh

Đồ án môn Hệ thống sinh trắc học: xây dựng hệ thống **xác thực vân tay 1:1 không tiếp xúc** (contactless fingerprint verification). Người dùng giơ ngón tay trước **camera điện thoại** thay vì đặt tay lên cảm biến, sau đó hệ thống crop phần đầu ngón tay có chứa vân. Hệ thống trích xuất điểm đặc trưng **minutiae** từ ảnh và so với mẫu đã đăng ký để trả lời câu hỏi: *"Ảnh này có phải cùng ngón tay với mẫu của người dùng X không?"*

### 1.2. Vì sao khó hơn vân tay tiếp xúc

- **Tương phản vân thấp:** ảnh chụp là da thật dưới ánh sáng, không phải vết ép lên kính, nên vân lồi và rãnh chỉ chênh nhau rất ít về độ sáng.
- **Cực tính ngược:** dưới đèn flash, vân lồi thường **sáng** và rãnh **tối**, ngược với ảnh cảm biến tiếp xúc (vân tối). Thuật toán vân tay kinh điển giả định vân tối, nên cần đảo ảnh.
- **Ánh sáng không đều:** flash tạo gradient sáng từ tâm ra rìa và các đốm phản xạ bão hòa trên da.
- **Tỷ lệ không cố định:** khoảng cách chụp thay đổi nên mật độ pixel trên vân thay đổi. Mọi ngưỡng (khoảng cách lọc minutiae, dung sai đối sánh) phụ thuộc tỷ lệ, nên phải chuẩn hóa về tương đương ~500 ppi (chu kỳ vân ≈ 9 px). Sai số ước lượng tỷ lệ giữa hai lần chụp vẫn còn vài phần trăm, nên matcher phải chịu được.
- **Méo phi tuyến:** ngón tay cong như hình trụ, vùng rìa bị nén lại khi chiếu lên ảnh; mỗi lần chụp góc nhìn lại khác. Chỉ vùng trung tâm đầu ngón là đáng tin.
- **Nền thay đổi:** nền có thể gần màu da (gỗ, tường màu be), phải định vị và tách ngón chính xác trước khi xử lý.

### 1.3. Hướng tiếp cận

Kết hợp học sâu cho bước định vị và xử lý ảnh cổ điển cho phần vân tay:

```text
ảnh điện thoại ─► detect (YOLO11n) ─► normalize ─► quality ─► enhance ─► skeleton ─► minutiae ─► Template
                                     (mask, xoay,                                                    │
                                      duỗi phẳng,          enroll: select_enrollment ─► lưu ◄────────┤
                                      chuẩn hóa tỷ lệ)     verify: match (MCC) với mọi mẫu ─► ≥ ngưỡng? ◄┘
```

### 1.4. Đã có sẵn

Model YOLO11n phát hiện đầu ngón tay (1 lớp `tip`) đã train xong. Kết quả train nằm ở `yolo11n_640/` (weights tại `yolo11n_640/weights/best.pt`; M0 sẽ sao chép sang `models/best.pt`). Train 100 epoch, `imgsz=640`, augment `fliplr=0.5`, `degrees=0`, `flipud=0`.

| Chỉ số | Valid | Test (460 ảnh, 473 đầu ngón) |
| --- | --- | --- |
| mAP50 | 0.988 | 0.970 |
| mAP50-95 | 0.708 | 0.636 |
| Precision | 0.966 | 0.968 |
| Recall | 1.000 | 0.947 |

Kết luận: đủ tốt cho bước crop. mAP50-95 thấp hơn nghĩa là biên box chưa thật sát, nên cần `pad` khi crop. Vì train không có augment xoay, ngón chĩa ngang/xuống có thể bị bỏ sót, nên detect có TTA xoay (mục 6.3). **Không train lại YOLO** trong phạm vi tài liệu này.

### 1.5. Mục tiêu đo được

Mọi chỉ số EER dưới đây báo cáo trên **tập test** (subject không dùng để tinh chỉnh, xem mục 8.2).

| Chỉ số | Tối thiểu | Mục tiêu | Mốc tham khảo |
| --- | --- | --- | --- |
| EER, chụp có kiểm soát, ngón trỏ | ≤ 20% | 10–15% | 10.71% (Priesnitz et al., Sensors 2022, dùng FingerNet + SourceAFIS) |
| EER, chụp không kiểm soát | báo cáo | ≤ 35% | 30.41% (cùng nguồn) |
| FTA (tỷ lệ thu nhận thất bại), có kiểm soát | báo cáo | ≤ 10% | — |
| Thời gian `extract()` / ảnh, laptop CPU | ≤ 5 s | ≤ 3 s | — |

Dự án dùng minutiae Crossing Number cổ điển (yếu hơn FingerNet), nên ngưỡng tối thiểu đặt cao hơn mốc tham khảo. Phần bù lại là matcher MCC và các bước bù méo (duỗi phẳng, làm phẳng ánh sáng).

### 1.6. Điều kiện chụp ảnh đầu vào

- Một ngón (mặt bụng ngón), bật flash, ảnh JPEG độ phân giải đầy đủ (~12 MP). Ảnh có thể mang thông tin xoay EXIF.
- **Khoảng cách 8–15 cm, chạm để lấy nét vào đầu ngón** (hoặc dùng zoom 2x nếu máy có camera tele). Camera chính của nhiều điện thoại không lấy nét được ở 5 cm.
- Ước tính độ phân giải: ở 10 cm, ảnh 12 MP (rộng 4000 px, góc nhìn ngang ~70°) cho khoảng 28 px/mm, tức chu kỳ vân ~13–14 px. Như vậy là đủ, và vẫn phải chuẩn hóa về 9 px.
- Các điều kiện này cần báo lại cho người làm ứng dụng di động (hướng dẫn trên màn hình chụp).

---

## 2. Phạm vi

**Trong phạm vi (agent làm):**

- Package Python `fpsys`: toàn bộ pipeline từ ảnh đến Template, đối sánh, quyết định enroll/verify, lưu trữ cục bộ dạng JSON để thử nghiệm.
- CLI cho workflow enroll / verify / extract / visualize.
- Script đánh giá: chia tập dev/test, sinh cặp, tính FAR/FRR/EER, đường DET, ablation, baseline ngoài (tùy chọn).
- Unit test bằng pytest.
- (M8, tùy chọn) embedding học sâu và kết hợp điểm.

**Ngoài phạm vi (không làm):**

- Backend FastAPI, cơ sở dữ liệu, mã hóa, xác thực API, ứng dụng Android (thành viên khác làm; họ chỉ import API công khai ở mục 5).
- Train hay sửa model YOLO.
- Chống giả mạo (presentation attack detection), nhận dạng 1:N, tái tạo 3D.

---

## 3. Môi trường và ràng buộc

- Python ≥ 3.10, chạy được trên laptop **chỉ có CPU** (Windows và Linux).
- Thư viện: `numpy`, `scipy`, `opencv-python` (hoặc `-headless`), `scikit-image`, `ultralytics`, `pyyaml`, `pillow`, `matplotlib`, `pytest`. Tùy chọn: `fingerprint-enhancer` (ablation), `onnxruntime` (M8).
- **Xung đột OpenCV:** `ultralytics` kéo theo `opencv-python`; cài thêm `opencv-contrib-python` sẽ xung đột namespace `cv2`. Vì vậy **không dùng `cv2.ximgproc`**. Làm mảnh vân bằng `skimage.morphology.skeletonize`. `cv2.grabCut` và `cv2.remap` nằm trong bản thường, dùng được.
- **Không commit** `data/`, `models/`, `results/`, `yolo11n_640/`, `*.pt`, `*.onnx`, ảnh người tham gia, dữ liệu RidgeBase. Thêm vào `.gitignore`. (`yolo11n_640/train_batch*.jpg` chứa ảnh của dataset YOLO.)
- Không lưu ảnh ra đĩa trong luồng chính; chỉ ghi ảnh trung gian khi người dùng truyền `debug_dir`.
- Kết quả phải tái lập được: mọi ngẫu nhiên có `seed`, mọi tham số nằm trong `configs/default.yaml`.
- Code có type hint đầy đủ, docstring ngắn, hàm thuần (nhận numpy, trả dataclass), dùng `logging` thay `print` trong package.

---

## 4. Cấu trúc repo

```text
Contactless_Fingerprint_Verification/
├── configs/
│   └── default.yaml          # toàn bộ tham số, xem mục 7
├── models/
│   ├── best.pt               # YOLO11n, sao chép từ yolo11n_640/weights/ (gitignore)
│   └── embed.onnx            # M8, tùy chọn (gitignore)
├── yolo11n_640/              # kết quả train YOLO, giữ làm hồ sơ (gitignore)
├── fpsys/
│   ├── __init__.py           # API công khai, xem mục 5
│   ├── config.py             # load YAML → dataclass Config, override "a.b=c", cfg_hash()
│   ├── types.py              # Minutia, Template, VerifyResult, CaptureError, ErrorCode, TemplateMismatchError
│   ├── io.py                 # load_image(path | bytes) có xử lý EXIF
│   ├── detect.py
│   ├── normalize.py          # mask, xoay, duỗi phẳng, chu kỳ vân, tỷ lệ
│   ├── quality.py
│   ├── enhance.py
│   ├── skeleton.py
│   ├── minutiae.py
│   ├── match_mcc.py          # Minutia Cylinder-Code + LSS-R (mặc định)
│   ├── match_hough.py        # Hough có scale (so sánh, ablation)
│   ├── matcher.py            # match(): chọn method theo config, kiểm tra cfg_hash
│   ├── decision.py           # verify(), select_enrollment()
│   ├── embed.py              # M8, tùy chọn
│   ├── pipeline.py           # extract(): ghép các bước
│   ├── store.py              # lưu template cục bộ dạng JSON (thử nghiệm, eval)
│   ├── viz.py                # vẽ ảnh trung gian, minutiae
│   └── cli.py                # python -m fpsys ...
├── eval/
│   ├── datasets.py           # đọc dữ liệu tự thu, FVC2002, RidgeBase
│   ├── splits.py             # chia subject dev/test
│   ├── make_pairs.py
│   ├── run_eval.py
│   ├── metrics.py
│   └── baselines/            # tùy chọn: SourceAFIS / NBIS
├── train_embed/              # M8: notebook train trên Colab (người dùng tự chạy)
├── tests/
├── notebooks/                # 01_crop, 02_enhance, 03_minutiae, 04_eval (người dùng tự chạy)
├── data/                     # gitignore
└── README.md
```

---

## 5. Hợp đồng dữ liệu (chốt trước, không đổi tùy tiện)

`fpsys/types.py`:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

class ErrorCode(str, Enum):
    NO_FINGER = "NO_FINGER"                  # YOLO không thấy đầu ngón (kể cả sau TTA)
    BLURRY = "BLURRY"                        # năng lượng dải tần vân dưới ngưỡng / không ước lượng được chu kỳ
    BAD_EXPOSURE = "BAD_EXPOSURE"            # quá tối / quá sáng / quá nhiều pixel bão hòa
    LOW_QUALITY = "LOW_QUALITY"              # vùng vân hợp lệ quá nhỏ sau enhance
    TOO_FEW_MINUTIAE = "TOO_FEW_MINUTIAE"
    INCONSISTENT_ENROLL = "INCONSISTENT_ENROLL"  # các mẫu enroll không khớp nhau

class CaptureError(Exception):
    def __init__(self, code: ErrorCode, detail: str = ""):
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail

class TemplateMismatchError(ValueError):
    """Hai template sinh từ cấu hình trích xuất khác nhau (cfg_hash khác)."""

@dataclass
class Minutia:
    x: float                                  # px, hệ tọa độ ảnh đã chuẩn hóa, gốc trên-trái, y hướng xuống
    y: float
    theta: float                              # radian, [0, 2π), atan2(dy, dx) trong hệ tọa độ ảnh
    kind: Literal["ending", "bifurcation"]
    quality: float                            # 0..1

@dataclass
class Template:
    minutiae: list[Minutia]
    shape: tuple[int, int]                    # (h, w) ảnh đã chuẩn hóa
    ridge_period: float                       # px sau chuẩn hóa, kỳ vọng ≈ 9
    version: str = "fp-v1"
    embedding: list[float] | None = None      # M8; None khi không bật fusion
    meta: dict = field(default_factory=dict)  # cfg_hash, conf YOLO, quality, scale, angle, timing_ms...

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Template": ...

@dataclass
class VerifyResult:
    score: float
    threshold: float
    accept: bool
    per_template: list[float]                 # điểm với từng mẫu đã enroll
```

**Quy ước góc minutiae** (bắt buộc, vì matcher không yêu cầu cùng loại):

- Ending: θ là hướng **từ trong thân vân đi ra điểm kết thúc**.
- Bifurcation: θ là hướng **từ điểm rẽ nhánh đi về phía nhánh đơn**.

Với quy ước này, một ending trên vân lồi trùng vị trí và gần trùng góc với bifurcation tương ứng trên rãnh (tính đối ngẫu). Khi ảnh nhiễu làm đổi loại minutia, góc vẫn khớp.

**`meta["cfg_hash"]`**: hash SHA-1 (8 ký tự đầu) của các mục config ảnh hưởng tới đặc trưng: `pipeline`, `normalize`, `enhance`, `skeleton`, `minutiae` (và `fusion.model` khi bật M8). `match()` raise `TemplateMismatchError` khi hai template có `cfg_hash` khác nhau. Đổi các mục này đồng nghĩa phải enroll lại.

API công khai (`fpsys/__init__.py`), backend chỉ dùng các hàm này:

```python
def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config: ...

def extract(image: np.ndarray, cfg: Config | None = None,
            debug_dir: str | None = None) -> Template:
    """Ảnh BGR uint8 → Template. Loại ảnh đầu vào theo cfg.pipeline.source. Lỗi chụp → raise CaptureError."""

def match(probe: Template, ref: Template, cfg: Config | None = None) -> float:
    """Điểm tương đồng trong [0, 1], đối xứng: match(a, b) == match(b, a)."""

def select_enrollment(templates: list[Template], cfg: Config | None = None) -> list[Template]:
    """Lọc các mẫu enroll không nhất quán. Còn ít hơn enroll.min_templates → CaptureError(INCONSISTENT_ENROLL)."""

def verify(probe: Template, refs: list[Template], cfg: Config | None = None) -> VerifyResult:
    """Gộp điểm với mọi mẫu đã enroll theo cfg.verify.fusion, so với cfg.verify.threshold."""
```

---

## 6. Đặc tả từng module

### 6.1. `io.py`

- `load_image(src: str | bytes) -> np.ndarray` trả ảnh BGR uint8.
- Đọc bằng Pillow, gọi `ImageOps.exif_transpose`, rồi chuyển sang BGR. Lý do: `cv2.imdecode` (dùng khi backend nhận bytes) **không** áp dụng xoay EXIF, nên ảnh điện thoại có thể bị nằm ngang.

### 6.2. `config.py`

- Load YAML thành dataclass lồng nhau; `load_config(overrides=["enhance.invert=false"])` ép kiểu theo kiểu của giá trị mặc định (bool, int, float, list).
- `cfg_hash(cfg) -> str` theo mục 5.
- `pipeline.source`:
  - `contactless`: ảnh điện thoại đầy đủ, chạy toàn bộ pipeline.
  - `contactless_crop`: ảnh đã là crop đầu ngón (ví dụ một số bộ dữ liệu công khai), bỏ qua detect.
  - `contact`: ảnh cảm biến tiếp xúc (FVC). Bỏ qua detect, kiểm tra phơi sáng, mask da, xoay, duỗi phẳng, đảo cực tính; mask lấy từ ngưỡng phương sai khối; không chuẩn hóa tỷ lệ (giả định đã ~500 ppi).

### 6.3. `detect.py`

- Nạp `YOLO(cfg.detect.weights)` **một lần** (cache theo đường dẫn), chạy trên ảnh gốc, `imgsz=cfg.detect.imgsz`, `conf=cfg.detect.conf`.
- Ultralytics trả tọa độ box theo **ảnh gốc**. Crop trực tiếp trên ảnh gốc độ phân giải đầy đủ, **không** crop từ ảnh đã resize.
- **Chọn box:** lấy các box có conf ≥ conf_max − `tie_conf_delta`, trong đó chọn box diện tích lớn nhất (ngón được chụp thường là ngón gần camera nhất).
- **TTA xoay:** nếu `tta_rotate` và không có box, chạy lại trên ảnh xoay 90° và 270°, đổi tọa độ box về ảnh gốc, crop trên ảnh gốc.
- Nới mỗi chiều thêm `pad` (mặc định 10%), kẹp trong biên ảnh.
- Vẫn không có box → `CaptureError(NO_FINGER)`.
- Trả về `Detection(crop, box_xyxy, conf, n_boxes, rotated_deg)`.

### 6.4. `normalize.py`

Input: crop BGR. Output: `Normalized(gray, mask, angle_deg, scale, ridge_period, period_raw, band_energy, sat_ratio, brightness)`.

1. **Mask da** (tính trên crop thu nhỏ về rộng `mask_work_width` px, rồi phóng mask về kích thước crop bằng nội suy nearest):
   - `grabcut` (mặc định): `cv2.grabCut` khởi tạo bằng hình chữ nhật là box YOLO trước khi pad (trong hệ tọa độ crop), `grabcut_iters` vòng.
   - `otsu_cr`: chuyển sang YCrCb, ngưỡng Otsu trên kênh Cr.
   - `fixed_ycrcb`: khoảng cố định Cr 133–173, Cb 77–127.
   - Sau đó: giữ thành phần liên thông lớn nhất, lấp lỗ, mở/đóng hình thái học, co lại `mask_erode_px`.
2. **Phơi sáng:** trên vùng mask, tính `brightness` = trung bình ảnh xám và `sat_ratio` = tỷ lệ pixel có kênh lớn nhất ≥ `sat_level`. Loại pixel bão hòa (nở thêm `sat_dilate_px`) khỏi mask, vì đốm phản xạ không chứa vân.
3. **Xoay:** PCA trên tọa độ pixel của mask cho trục chính của ngón. Xoay để trục thẳng đứng. Xác định đầu nào là **đỉnh ngón**, theo thứ tự:
   1. Đầu mà mask *không* chạm biên crop (phía gốc ngón chạm biên vì box cắt ngang thân ngón).
   2. Nếu cả hai đầu cùng chạm hoặc cùng không chạm biên (ảnh chụp cận): so bề rộng mask trung bình trong đoạn `tip_end_frac` cuối mỗi đầu; đầu hẹp hơn (bo tròn) là đỉnh.

   Xoay thêm 180° nếu cần để đỉnh ngón hướng lên.
4. **Duỗi phẳng (nếu `unwarp`):** mô hình ngón là hình trụ có trục thẳng đứng. Với mỗi hàng y, lấy biên trái/phải của mask → tâm c(y), bán kính r(y); làm mượt c, r theo y bằng median filter. Tọa độ duỗi phẳng x' = r·arcsin((x − c)/r) (độ dài cung). Thực hiện bằng `cv2.remap` với ánh xạ ngược x = c + r·sin(x'/r). Pixel có |x − c|/r > sin(`unwarp_max_angle_deg`) bị loại khỏi mask, vì vùng đó nén quá mạnh.
5. **Ước lượng chu kỳ vân:** trên vùng trung tâm mask, chia các khối `period_block`×`period_block` nằm hoàn toàn trong mask. Mỗi khối: trừ trung bình, nhân cửa sổ Hann, lấy FFT 2D, tìm đỉnh năng lượng theo bán kính trong dải ứng với chu kỳ `[period_min, period_max]` px. Chu kỳ = `period_block` / bán kính đỉnh. Đồng thời tính `band_energy` của khối = năng lượng trong vành ±1 bin quanh đỉnh / tổng năng lượng (bỏ thành phần DC). Lấy **trung vị** chu kỳ và `band_energy` các khối hợp lệ. Không có khối hợp lệ nào → `CaptureError(BLURRY)`.
6. **Chuẩn hóa tỷ lệ:** `scale = target_period / period_raw`, kẹp trong `[scale_min, scale_max]`, resize ảnh xám và mask (mask dùng nội suy nearest). `ridge_period` = chu kỳ sau resize.
7. Ảnh xám lấy theo `cfg.enhance.channel`: `"gray"` hoặc `"green"` (kênh G thường có tương phản vân da tốt hơn; đánh giá cả hai bằng ablation).

### 6.5. `quality.py`

- `assess(norm: Normalized, cfg) -> QualityReport(sharpness, brightness, sat_ratio, laplacian_var, ok, code)`. Chạy **sau normalize**, chỉ trên vùng mask.
- `sharpness` = `norm.band_energy`. Đây là chỉ số độ nét đặc thù cho vân: ảnh mờ mất năng lượng ở dải tần vân trước tiên, còn viền móng, nền hay nhiễu JPEG thì không làm tăng nó.
- `laplacian_var` (phương sai Laplacian trên vùng mask, sau khi resize về rộng 512 px) chỉ ghi vào meta để so sánh trong báo cáo, không dùng để chặn.
- Thứ tự kiểm tra: `brightness` ngoài `[min_brightness, max_brightness]` hoặc `sat_ratio > max_sat_ratio` → `BAD_EXPOSURE`; `sharpness < min_band_energy` → `BLURRY`.
- Ngưỡng trong config (giá trị khởi đầu ở mục 7) **phải được hiệu chỉnh** trên ảnh thật; đừng coi giá trị mặc định là đúng.

### 6.6. `enhance.py`

Input: `Normalized`. Output: `Enhanced(image, orientation, frequency, valid_mask, coherence, flat)`. Theo phương pháp Hong–Wan–Jain (1998), thêm bước làm phẳng ánh sáng.

1. **Làm phẳng ánh sáng** (nếu `flatten_sigma > 0`): `flat = gray / (GaussianBlur(gray, σ = flatten_sigma × target_period) + ε)`, chỉ tính trong mask (dùng normalized convolution: blur(gray·mask) / blur(mask) để biên mask không bị tối). Kéo về [0, 255].
2. CLAHE (`clip_limit`, `tile`); `clip_limit = 0` nghĩa là bỏ qua.
3. Đảo cực tính nếu `cfg.enhance.invert` (mặc định `true`, để vân thành tối như ảnh tiếp xúc). **Kiểm chứng bằng mắt** trên vài ảnh thật; nếu sai thì đổi config, không sửa code.
4. Chuẩn hóa mean/variance theo khối (`block` px) trong mask.
5. **Trường hướng:** gradient Sobel → Gxx, Gyy, Gxy, làm mượt Gaussian (`orient_sigma`). θ = ½·atan2(2Gxy, Gxx − Gyy) + π/2. Tính độ nhất quán (coherence) để đánh giá chất lượng khối.
6. **Tần số vân** theo khối 32×32: xoay khối theo hướng vân, lấy x-signature, đo khoảng cách giữa các đỉnh. Khối không hợp lệ thì dùng trung vị toàn ảnh. Sau chuẩn hóa tỷ lệ, giá trị kỳ vọng ≈ 1/9.
7. **Lọc Gabor định hướng:** lượng tử hướng thành `n_angles` góc, tạo bank kernel bằng `cv2.getGaborKernel` với bước sóng = 1/tần số và σ = `gabor_k` × bước sóng. Mỗi pixel lấy đáp ứng của kernel đúng hướng khối.
8. `valid_mask` = mask ∩ (coherence > `min_coherence`). Nếu `valid_mask` nhỏ hơn `min_valid_area` → `CaptureError(LOW_QUALITY)`.

Viết hàm `enhance_reference()` gọi thư viện `fingerprint-enhancer` (nếu đã cài) để làm mốc so sánh trong ablation. Hàm này không thuộc luồng chính.

### 6.7. `skeleton.py`

1. Nhị phân hóa: vân = đáp ứng Gabor < 0 (dấu phụ thuộc cực tính; kiểm tra), chỉ trong `valid_mask`.
2. Bỏ các thành phần nhỏ hơn `min_component_px`; lấp lỗ nhỏ hơn `max_hole_px`.
3. `skimage.morphology.skeletonize`.
4. **Cắt nhánh gai:** từ mỗi điểm kết thúc, dò theo skeleton tối đa `spur_len_px` px. Nếu gặp điểm rẽ nhánh trước khi hết độ dài thì xóa đoạn vừa dò. Lặp 2 lần.
5. Bỏ các đoạn skeleton rời có độ dài nhỏ hơn `min_segment_px`.

### 6.8. `minutiae.py`

1. **Crossing Number** trên lân cận 8: CN = ½·Σ|Pᵢ − Pᵢ₊₁| (vòng tròn 8 điểm). CN = 1 là ending, CN = 3 là bifurcation.
2. Bỏ các điểm nằm ngoài `valid_mask` đã co thêm `border_px`.
3. **Góc** theo quy ước ở mục 5: ending dò theo vân `trace_px` px, góc là hướng từ điểm dò được về minutia. Bifurcation lấy hướng từ trường hướng, chọn chiều (θ hay θ + π) hướng về nhánh đơn. Chuẩn hóa về [0, 2π).
4. **Lọc minutiae giả:**
   - Hai ending cách nhau < `break_px` và hướng gần đối nhau (vân bị đứt) → bỏ cả hai.
   - Hai bifurcation cách nhau < `bridge_px` (cầu nối hoặc lỗ) → bỏ cả hai.
   - Ending và bifurcation cách nhau < `spur_px` → bỏ cả hai.
   - Cụm quá dày (nhiều hơn 4 điểm trong bán kính `cluster_px`) → bỏ cả cụm.
5. `quality` = coherence trung bình quanh điểm.
6. Ít hơn `min_minutiae` điểm → `CaptureError(TOO_FEW_MINUTIAE)`.

### 6.9. Matcher

`matcher.py` cung cấp `match()`: kiểm tra `cfg_hash` (khác nhau → `TemplateMismatchError`), một trong hai template có ít hơn 3 minutiae → 0, rồi gọi method theo `cfg.matcher.method`. Kết quả phải đối xứng; method nào không đối xứng tự nhiên thì lấy trung bình hai chiều.

#### 6.9.1. `match_mcc.py` — Minutia Cylinder-Code (mặc định)

Theo Cappelli, Ferrara, Maltoni (TPAMI 2010). Đây là mô tả cục bộ, không cần căn chỉnh toàn cục, nên chịu được méo phi tuyến và sai số tỷ lệ nhỏ tốt hơn Hough.

1. **Cylinder:** với mỗi minutia, dựng lưới `ns`×`ns` ô không gian trong bán kính `radius`, xoay theo θ của minutia, × `nd` ô góc. Giá trị mỗi ô = Σ (đóng góp không gian Gaussian `sigma_s`) × (đóng góp góc Gaussian `sigma_d` của chênh lệch hướng) trên các minutia lân cận, qua hàm sigmoid (`mu_psi`, `tau_psi`).
2. **Ô hợp lệ:** ô nằm trong bao lồi của các minutiae nới thêm `hull_offset` px. Cylinder không hợp lệ nếu tỷ lệ ô hợp lệ < `min_valid_cells` hoặc có ít hơn `min_neighbors` minutia đóng góp.
3. **Độ tương đồng cục bộ** giữa hai cylinder: 1 − ‖a − b‖ / (‖a‖ + ‖b‖) trên các ô hợp lệ ở cả hai; bằng 0 nếu chênh θ giữa hai minutia gốc > `max_dir_diff_deg`. Tính cả ma trận n_probe × n_ref bằng phép nhân ma trận numpy.
4. **Gộp điểm toàn cục bằng LSS-R:** số cặp dùng để tính điểm nP = `min_np` + round(sigmoid(min(n_probe, n_ref); `mu_p`, `tau_p`) × (`max_np` − `min_np`)). Chọn các cặp có độ tương đồng cục bộ cao nhất, chạy relaxation `relax.n_iter` vòng với trọng số `relax.w` (phạt các cặp không tương thích nhau về khoảng cách và góc tương đối, tham số `relax.mu*` / `relax.tau*`), rồi lấy trung bình nP cặp tốt nhất sau relaxation.
5. Điểm trong [0, 1]. Đối chiếu lại các tham số mặc định với bài báo gốc trước khi chốt. Tham số không gian (`radius`, `sigma_s`, `hull_offset`) tính theo 500 ppi, khớp với `target_period = 9`.
6. Mục tiêu < 50 ms cho hai template mỗi bên khoảng 60 điểm (cylinder có thể tính sẵn một lần khi load template).

#### 6.9.2. `match_hough.py` — Hough có scale (so sánh)

1. **Bỏ phiếu:** với mỗi cặp (p ∈ probe, r ∈ ref) và mỗi s ∈ `scales`: Δθ = θr − θp, bỏ qua nếu |Δθ| > `max_rotation_deg` (cả hai ảnh đã được xoay ngón về thẳng đứng nên góc quay thật nhỏ). Xoay p quanh gốc tọa độ một góc Δθ, nhân s, tính (Δx, Δy) = r − p'. Lượng tử Δθ theo `theta_step_deg`, Δx và Δy theo `xy_step_px`, cộng phiếu (trọng số `quality`).
2. Lấy `top_k` phép biến đổi nhiều phiếu nhất. Với mỗi phép: biến đổi toàn bộ probe, **ghép cặp một-một tham lam** theo khoảng cách tăng dần, điều kiện khoảng cách ≤ `match_dist_px` và chênh góc (vòng tròn) ≤ `match_angle_deg`. Không bắt buộc cùng loại.
3. Điểm = m² / (n_probe × n_ref), kẹp trong [0, 1], lấy lớn nhất trong `top_k` phép, rồi trung bình hai chiều.

### 6.10. `decision.py`

- `select_enrollment(templates, cfg)`: tính `match` chéo mọi cặp; với mỗi template lấy trung vị điểm với các template còn lại; loại template có trung vị < `enroll.min_consistency`. Còn ít hơn `enroll.min_templates` → `CaptureError(INCONSISTENT_ENROLL)`. Ghi log template bị loại.
- `verify(probe, refs, cfg)`: tính `match(probe, r)` với mọi r; gộp theo `verify.fusion`: `max` (mặc định) hoặc `mean_top2`; so với `verify.threshold`.

### 6.11. `pipeline.py`

```python
def extract(image, cfg=None, debug_dir=None) -> Template:
    cfg = cfg or load_config()
    det = None
    if cfg.pipeline.source == "contactless":
        det = detect(image, cfg)                 # NO_FINGER
        crop = det.crop
    else:                                        # contactless_crop | contact
        crop = image
    norm = normalize(crop, cfg, det)             # BLURRY nếu không ước lượng được chu kỳ
    q    = assess(norm, cfg)                     # BAD_EXPOSURE / BLURRY
    enh  = enhance(norm, cfg)                    # LOW_QUALITY
    sk   = skeletonize(enh, cfg)
    mins = extract_minutiae(sk, enh, cfg)        # TOO_FEW_MINUTIAE
    emb  = embed(norm, cfg) if cfg.fusion.enabled else None   # M8
    return Template(mins, norm.gray.shape, norm.ridge_period, embedding=emb,
                    meta={"cfg_hash": cfg_hash(cfg), "conf": det.conf if det else None,
                          "quality": asdict(q), "scale": norm.scale, "angle_deg": norm.angle_deg,
                          "period_raw": norm.period_raw, "timing_ms": {...}})
```

- Đo thời gian từng bước, lưu vào `meta["timing_ms"]`.
- Có `debug_dir` thì lưu PNG từng bước: `01_crop`, `02_mask`, `03_norm` (đã xoay, duỗi phẳng, chuẩn hóa tỷ lệ), `04_flat`, `05_enhanced`, `06_orientation`, `07_binary`, `08_skeleton`, `09_minutiae`.
- Chế độ `contact` dùng để phát triển và kiểm thử skeleton, minutiae, matcher trên FVC trước khi có đủ ảnh không tiếp xúc tốt.

### 6.12. `store.py` (cục bộ, phục vụ CLI và đánh giá)

- File JSON đặt trong `cfg.store.path`, cấu trúc `{user_id: {finger: [template_dict, ...]}}`.
- `enroll(user_id, finger, templates)` gọi `select_enrollment` rồi lưu; `get(user_id, finger)`; `delete(user_id)`.
- `verify(user_id, finger, probe)` gọi `decision.verify` với các template đã lưu.
- Backend thật sẽ thay file JSON bằng SQLite có mã hóa; logic quyết định nằm ở `decision.py` nên dùng chung.

### 6.13. `cli.py`

```bash
python -m fpsys extract IMG [--debug-dir out/] [--set enhance.invert=false]
python -m fpsys enroll --user U01 --finger R2 IMG1 IMG2 IMG3
python -m fpsys verify --user U01 --finger R2 IMG
python -m fpsys match IMG_A IMG_B              # in điểm, tiện để thử nhanh
python -m fpsys viz IMG --out out.png          # lưới 3×3 ảnh 9 bước, dùng cho báo cáo
```

Mã thoát: 0 khi chấp nhận hoặc thành công, 1 khi từ chối, 2 khi có `CaptureError` (in mã lỗi).

---

## 7. `configs/default.yaml` (giá trị khởi đầu, sẽ hiệu chỉnh)

```yaml
pipeline: {source: contactless}          # contactless | contactless_crop | contact
detect:   {weights: models/best.pt, conf: 0.5, pad: 0.10, imgsz: 640, tie_conf_delta: 0.10, tta_rotate: true}
normalize:
  skin_method: grabcut                   # grabcut | otsu_cr | fixed_ycrcb
  mask_work_width: 384
  grabcut_iters: 5
  mask_erode_px: 6
  sat_level: 250
  sat_dilate_px: 5
  tip_end_frac: 0.15
  unwarp: true
  unwarp_max_angle_deg: 65
  target_period: 9.0
  period_block: 64
  period_min: 4
  period_max: 30
  scale_min: 0.25
  scale_max: 3.0
quality:  {min_brightness: 40, max_brightness: 220, max_sat_ratio: 0.15, min_band_energy: 0.20}
enhance:
  channel: gray                          # gray | green
  invert: true
  flatten_sigma: 2.5                     # bội số của target_period; 0 = tắt
  clahe: {clip_limit: 2.0, tile: 8}
  block: 16
  orient_sigma: 7.0
  gabor_k: 0.65
  n_angles: 16
  min_coherence: 0.3
  min_valid_area: 0.15                   # tỷ lệ so với diện tích ảnh chuẩn hóa
skeleton: {min_component_px: 30, max_hole_px: 20, spur_len_px: 10, min_segment_px: 15}
minutiae:
  border_px: 12
  trace_px: 10
  break_px: 10
  bridge_px: 8
  spur_px: 8
  cluster_px: 10
  min_minutiae: 12
matcher:
  method: mcc                            # mcc | hough
  mcc:
    radius: 70
    ns: 16
    nd: 6
    sigma_s: 9.333                       # 28/3
    sigma_d: 0.698                       # 2π/9
    mu_psi: 0.01
    tau_psi: 400
    hull_offset: 50
    min_valid_cells: 0.75
    min_neighbors: 2
    max_dir_diff_deg: 90
    min_np: 4
    max_np: 12
    mu_p: 20
    tau_p: 0.4
    relax: {n_iter: 5, w: 0.5, mu1: 5.0, tau1: -1.6, mu2_deg: 15, tau2: -30, mu3_deg: 15, tau3: -30}
  hough:
    theta_step_deg: 10
    xy_step_px: 8
    scales: [0.90, 0.95, 1.00, 1.05, 1.10]
    max_rotation_deg: 45
    top_k: 3
    match_dist_px: 15
    match_angle_deg: 20
enroll:   {min_templates: 2, min_consistency: 0.15}
verify:   {threshold: 0.10, fusion: max}   # max | mean_top2; threshold thay bằng giá trị từ metrics.json (tập dev)
fusion:   {enabled: false, model: models/embed.onnx, w: 0.5}   # M8
store:    {path: data/store.json}
eval:     {dev_fraction: 0.4, target_far: 0.01, n_bootstrap: 1000}
seed: 42
```

`config.py` hỗ trợ ghi đè từ CLI và script eval: `--set enhance.invert=false matcher.method=hough`.

---

## 8. Workflow đánh giá (`eval/`)

### 8.1. Dữ liệu

- **Tự thu** (`data/own/`), tên file `{subject}_{finger}_{session}_{cond}_{sample}.jpg`, ví dụ `S03_R2_s1_c_04.jpg`:
  - `finger`: R2 (trỏ phải), L2 (trỏ trái); có thể thêm R3, L3.
  - `session`: `s1`, `s2`, chụp cách nhau **ít nhất 1 ngày**.
  - `cond`: `c` (có kiểm soát: đúng mục 1.6, nền tối trơn) hoặc `u` (không kiểm soát: nền và ánh sáng tùy ý, cầm máy tự nhiên).
  - Enroll luôn dùng `s1_c`. Probe `s2_c` cho EER có kiểm soát, `s2_u` cho EER không kiểm soát.
  - Giao thức tối thiểu: **≥ 20 người × 2 ngón × 2 phiên × 5 ảnh (c)**, cộng 3 ảnh (u) ở phiên 2. Dùng cùng một điện thoại và ghi lại model máy. Có phiếu đồng ý của người tham gia (dữ liệu sinh trắc học).
  - Với khoảng 20 người, khoảng tin cậy của EER sẽ rộng; phải nêu rõ điều này trong báo cáo.
- **RidgeBase** (`data/ridgebase/`): bộ contactless chụp bằng điện thoại, dùng làm benchmark contactless chính để so sánh với các công bố khác. **Agent phải xem cấu trúc thư mục thực tế và hỏi người dùng trước khi viết parser.** Nếu ảnh đã là crop từng ngón thì dùng `pipeline.source=contactless_crop`. Dùng giao thức/chia tập chính thức nếu bộ dữ liệu có.
- **FVC2002 DB1_B** (vân tay tiếp xúc, tải tự do): `data/fvc2002_db1b/101_1.tif`; 10 ngón × 8 lần lấy mẫu. Dùng với `pipeline.source=contact`, chỉ để gỡ lỗi nửa sau pipeline.
- `datasets.py` parse tên file thành bản ghi `(path, subject, finger, session, cond, sample)`. Tên file không hợp lệ thì báo lỗi rõ ràng, không bỏ qua im lặng.

### 8.2. Chia tập, sinh cặp và chấm điểm

- `splits.py`: chia **theo subject** thành dev (`eval.dev_fraction`) và test, có seed, ghi ra `data/splits/{dataset}.json` để cố định. FVC không chia.
  - Mọi tinh chỉnh (ngưỡng quality, chọn phương án ablation, ngưỡng verify, tham số fusion) **chỉ** dùng tập dev.
  - Tập test chỉ chạy với cấu hình đã chốt, để báo cáo con số cuối cùng.
- `make_pairs.py`:
  - **Genuine:** mọi cặp (enroll `s1_c`, probe `s2_{cond}`) cùng `(subject, finger)`.
  - **Impostor:** probe `s2_{cond}` với enroll `s1_c` của mọi `(subject, finger)` khác. Gắn nhãn phụ `same_subject` khi cùng người khác ngón.
  - Với FVC: genuine là mọi cặp lần lấy mẫu của cùng ngón; impostor là lần lấy mẫu đầu tiên của các ngón khác nhau (theo giao thức FVC).
- `run_eval.py --dataset own --split dev|test --cond c|u`:
  - Gọi `extract()` mỗi ảnh **một lần**, cache template vào `data/cache/{hash_ảnh}_{cfg_hash}.json`.
  - Ảnh lỗi `CaptureError` được ghi nhận (kèm mã lỗi) để tính FTA, không làm dừng chương trình.
  - Ghi `results/{run_name}/scores.csv` (probe, ref, label, same_subject, score), `metrics.json`, `det.png`, `hist.png`, `fta.csv`, `config.yaml` (bản sao config đã dùng).

### 8.3. Chỉ số (`metrics.py`)

- FAR(t), FRR(t) trên lưới ngưỡng.
- EER bằng nội suy tại điểm FAR = FRR.
- FRR tại FAR = `target_far` (1%).
- FTA theo từng mã lỗi; FRR tổng quát = FTA + (1 − FTA) × FRR.
- EER riêng cho tập impostor `same_subject`.
- Khoảng tin cậy 95% của EER: bootstrap **theo subject** (lấy mẫu lại subject, không lấy mẫu lại cặp), `n_bootstrap` lần, có seed.
- Vẽ đường DET (trục lệch chuẩn hóa) và histogram điểm genuine/impostor.
- Ngưỡng (chỉ tính trên tập dev): `threshold_eer` và `threshold_at_far` (tại `target_far`), ghi vào `metrics.json`. **Ngưỡng khuyến nghị cho `verify.threshold` là `threshold_at_far`**, vì hệ thống xác thực cần kiểm soát FAR. Khi chạy trên tập test, báo cáo cả FAR/FRR thực tế tại ngưỡng đã chọn từ dev.

### 8.4. Ablation (chỉ trên tập dev)

`run_eval.py --ablation`. Mỗi lần chạy chỉ đổi một yếu tố so với baseline, kết quả gom vào `results/ablation.csv` (EER, CI, FTA, thời gian trung bình):

| Tên | Thay đổi |
| --- | --- |
| baseline | `default.yaml` |
| hough | `matcher.method=hough` |
| no_unwarp | `normalize.unwarp=false` |
| no_flatten | `enhance.flatten_sigma=0` |
| otsu_mask | `normalize.skin_method=otsu_cr` |
| no_clahe | `enhance.clahe.clip_limit=0` |
| no_scale_norm | không resize, giữ `scale=1` |
| green | `enhance.channel=green` |
| no_invert | `enhance.invert=false` |
| ref_enhancer | thay `enhance` bằng `enhance_reference()` |
| otsu_crop | bỏ YOLO, phân đoạn da Otsu trên toàn ảnh để crop |
| mean_top2 | `verify.fusion=mean_top2` |

### 8.5. Baseline ngoài (tùy chọn, `eval/baselines/`)

Mục đích: tách lỗi do chất lượng ảnh với lỗi do matcher. Nếu baseline ngoài cũng cho EER cao trên cùng ảnh thì vấn đề nằm ở ảnh và enhance, không phải ở matcher.

- `export_enhanced.py`: xuất `Enhanced.image` (đã chuẩn hóa ~500 ppi, vân tối) thành PNG.
- **SourceAFIS** (giống bài Priesnitz, so sánh trực tiếp được): agent viết một chương trình Java nhỏ đọc `pairs.csv`, xuất `scores.csv`; người dùng tự cài JDK và tải jar SourceAFIS.
- **NBIS** (`mindtct` + `bozorth3`): chỉ khi người dùng đã có bản build (thường trên Linux/WSL).
- Điểm từ baseline đi qua cùng `metrics.py`.

### 8.6. Thời gian

`run_eval.py --timing`: chạy 50 ảnh, in thời gian trung bình và p95 của từng bước (lấy từ `meta["timing_ms"]`) và của `match`.

---

## 9. Milestone cho agent

Mỗi milestone kết thúc bằng: test xanh, `ruff` sạch, README cập nhật cách chạy.

### M0. Khung dự án

- Tạo cấu trúc mục 4, `pyproject.toml`, `.gitignore` (mục 3). Sao chép `yolo11n_640/weights/best.pt` → `models/best.pt`.
- Viết `types.py`, `config.py` (kể cả `cfg_hash`), `io.py`.
- **Xong khi:** `pytest` chạy được; `Template.to_dict` / `from_dict` cho lại đúng như ban đầu (kể cả `embedding=None` và có giá trị); ghi đè config từ chuỗi `a.b=c` đúng kiểu; `cfg_hash` đổi khi đổi `enhance.invert`, không đổi khi đổi `verify.threshold`.

### M1. Detect

- `detect.py`.
- **Xong khi:** ảnh màu đồng nhất raise `NO_FINGER`; test với YOLO giả (mock trả box cố định) xác nhận chọn box theo `tie_conf_delta` + diện tích, và TTA đổi tọa độ box về ảnh gốc đúng (sai ≤ 1 px). **Người kiểm tra:** chạy `viz` trên 20 ảnh thật, xác nhận crop đúng đốt ngón xa.

### M2. Normalize và quality

- `normalize.py`, `quality.py`.
- **Xong khi:**
  - Ảnh vân sin tổng hợp có chu kỳ đã biết (6, 9, 14 px, xoay ngẫu nhiên): chu kỳ ước lượng sai < 10%; sau chuẩn hóa chu kỳ trong khoảng 8–10 px.
  - Mask tổng hợp hình elip nghiêng 30° được xoay về thẳng đứng, đỉnh hướng lên; trường hợp elip chạm biên cả hai đầu (ảnh cận) vẫn chọn đúng đỉnh nhờ quy tắc bề rộng.
  - **Duỗi phẳng:** vân thẳng đứng cách đều dán lên hình trụ rồi chiếu vuông góc (bán kính đã biết): sau `unwarp`, hệ số biến thiên khoảng cách giữa các vân < 10% (trước khi duỗi phải > 20%).
  - `band_energy` giảm đơn điệu khi làm mờ Gaussian tăng dần; ảnh có 30% pixel bão hòa → `BAD_EXPOSURE`.
  - Ảnh nền gần màu da: `grabcut` cho IoU mask ≥ 0.9 với đáp án trên ảnh tổng hợp.

### M3. Enhance

- **Xong khi:** ảnh vân sin có thêm nhiễu Gaussian và gradient sáng tuyến tính: trường hướng sai trung bình < 10° so với hướng thật; ảnh sau Gabor có tương quan với ảnh sạch > 0.8; tắt `flatten_sigma` thì tương quan thấp hơn khi bật. **Người kiểm tra:** duyệt bằng mắt `04_flat` và `05_enhanced` trên ảnh thật, quyết định giá trị `invert` và `channel`.

### M4. Skeleton và minutiae (phát triển trên FVC trước)

- **Xong khi:**
  - Skeleton tổng hợp vẽ tay có đúng 1 ending và 1 bifurcation: tìm đủ 2 điểm, đúng loại, sai vị trí ≤ 2 px.
  - Skeleton luôn rộng đúng 1 px (không còn khối 2×2 điểm vân).
  - Đường thẳng có nhánh gai 5 px: gai bị cắt.
  - **Đối ngẫu:** ảnh nhị phân có một ending; lấy skeleton của phần bù (rãnh) cho một bifurcation cách ≤ 3 px và chênh góc ≤ 20° so với ending.
  - Trên FVC: trung bình 20–80 minutiae mỗi ảnh.

### M5. Matcher

- `match_mcc.py`, `match_hough.py`, `matcher.py`.
- **Xong khi** (áp dụng cho cả hai method, trừ khi ghi rõ):
  - `match(a, b) == match(b, a)` (sai ≤ 1e-6).
  - Template so với chính nó: Hough = 1.0, MCC ≥ 0.95.
  - So với bản xoay 15° và dịch (20, −10) px: ≥ 0.9.
  - So với bản xoay, dịch, **phóng 1.05**: MCC ≥ 0.7, Hough ≥ 0.6.
  - So với bản xoay, dịch, thêm 20% điểm nhiễu, bỏ 20% điểm, dịch mỗi điểm ngẫu nhiên ±3 px: ≥ 0.5.
  - So với template ngẫu nhiên cùng số điểm: < 0.2.
  - Hai template khác `cfg_hash` → `TemplateMismatchError`.
  - Hai template 60 điểm: < 50 ms.
  - Trên FVC2002 DB1_B: EER < 15% với cả hai method (pipeline cổ điển trên vân tay tiếp xúc nên đạt được dễ dàng; nếu không đạt là có lỗi); ghi lại EER của MCC so với Hough.

### M6. Pipeline, decision, store, CLI

- **Xong khi:** `extract` chạy hết trên toàn bộ ảnh tự thu mà không có exception nào ngoài `CaptureError`; thời gian trung bình ≤ 3 s/ảnh trên CPU; enroll rồi verify cùng ảnh thì chấp nhận; `select_enrollment` loại đúng một template lạc (của ngón khác) trong 4 template; `python -m fpsys viz` sinh lưới 9 bước.

### M7. Đánh giá

- **Xong khi:** `splits.py` sinh file chia tập cố định; `run_eval.py` sinh đủ `scores.csv`, `metrics.json` (EER, CI 95%, FRR@FAR=1%, FTA theo mã lỗi, FRR tổng quát, EER same_subject, `threshold_eer`, `threshold_at_far`), `det.png`, `hist.png` cho dữ liệu tự thu (dev và test, c và u), RidgeBase và FVC; `--ablation` sinh `ablation.csv` trên dev. Baseline ngoài (mục 8.5) làm nếu người dùng đã cài công cụ.

### M8. (Tùy chọn) Embedding học sâu và kết hợp điểm

Chỉ làm khi người dùng yêu cầu, sau khi M7 có kết quả baseline.

- **Train** (`train_embed/`, notebook chạy trên Colab có GPU, do người dùng chạy; được phép tải dữ liệu và trọng số ImageNet ở đó):
  - Input: crop 256×256 của `Normalized.gray` sau làm phẳng và CLAHE, quanh tâm vùng mask, ngoài mask điền 0. Không dùng ảnh Gabor vì nó có thể "bịa" vân.
  - Mạng: ResNet-18 khởi tạo ImageNet, embedding 256 chiều chuẩn hóa L2, loss ArcFace (s = 30, m = 0.5); lớp = (subject, finger).
  - Augment: xoay ±15°, phóng 0.9–1.1, độ sáng/tương phản, làm mờ nhẹ.
  - Dữ liệu train: RidgeBase (+ bộ contactless khác nếu có), subject **tách biệt hoàn toàn** với tập test của mục 8.2.
  - Xuất ONNX → `models/embed.onnx`.
- **Suy luận** (`fpsys/embed.py`): `onnxruntime` trên CPU, nạp model một lần; `extract` điền `Template.embedding` khi `fusion.enabled`. Điểm embedding = (cos + 1) / 2.
- **Kết hợp điểm:** chuẩn hóa z-score từng điểm (minutiae, embedding) bằng trung bình/độ lệch chuẩn của điểm impostor trên tập dev, s = w·z_m + (1 − w)·z_e, đưa về [0, 1] bằng sigmoid. `w` chọn bằng grid search trên dev; các tham số chuẩn hóa ghi vào config mục `fusion`.
- **Xong khi:** có EER trên tập test của ba cấu hình (minutiae, embedding, fusion) trong cùng một bảng; thời gian `extract` vẫn ≤ 3 s. Báo cáo trung thực nếu fusion không tốt hơn.

---

## 10. Quy tắc cho agent

- Không đổi chữ ký các hàm công khai hay các dataclass ở mục 5 nếu chưa hỏi người dùng; backend phụ thuộc vào chúng.
- Không tự "chỉnh" ngưỡng trong config để test pass. Test dùng ảnh tổng hợp có đáp án; ngưỡng cho ảnh thật do người dùng hiệu chỉnh dựa trên ảnh debug và tập dev.
- Không dùng tập test để chọn tham số. Mọi quyết định dựa trên tập dev.
- Khi một bước cần đánh giá bằng mắt (cực tính, chất lượng crop, mask, ảnh tăng cường), dừng lại, sinh ảnh debug và báo người dùng kiểm tra, thay vì đoán.
- Khi định dạng bộ dữ liệu ngoài (RidgeBase) chưa rõ, xem thư mục thực tế và hỏi người dùng trước khi viết parser.
- Không tải dữ liệu hay trọng số từ mạng, không đọc hay ghi ngoài thư mục repo (ngoại lệ duy nhất: notebook M8 do người dùng tự chạy trên Colab).
- Ưu tiên numpy vector hóa; vòng lặp Python theo pixel chỉ chấp nhận ở bước dò vân (trace) và cắt nhánh gai.
- Mỗi module có test riêng trong `tests/test_<module>.py`, dùng ảnh tổng hợp sinh bằng code (không cần file dữ liệu thật để chạy test). Test cần YOLO thì dùng mock.

---

## 11. Tài liệu tham khảo

- L. Hong, Y. Wan, A. Jain, "Fingerprint Image Enhancement: Algorithm and Performance Evaluation," *IEEE TPAMI*, 20(8), 1998: chuẩn hóa, trường hướng, tần số, lọc Gabor.
- R. Cappelli, M. Ferrara, D. Maltoni, "Minutia Cylinder-Code: A New Representation and Matching Technique for Fingerprint Recognition," *IEEE TPAMI*, 32(12), 2010: MCC và LSS-R.
- J. Priesnitz, R. Huesmann, C. Rathgeb, N. Buchmann, C. Busch, "Mobile Contactless Fingerprint Recognition: Implementation, Performance and Usability Aspects," *Sensors*, 22(3):792, 2022: mốc EER cho hệ thống smartphone.
- B. Jawade et al., "RidgeBase: A Cross-Sensor Multi-Finger Contactless Fingerprint Dataset," *IJCB*, 2022: bộ dữ liệu contactless.
- J. Deng, J. Guo, N. Xue, S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *CVPR*, 2019: loss cho embedding (M8).
- J. Engelsma, K. Cao, A. Jain, "Learning a Fixed-Length Fingerprint Representation," *IEEE TPAMI*, 2021 (DeepPrint): tham khảo cho embedding vân tay.
- D. Maltoni, D. Maio, A. Jain, S. Prabhakar, *Handbook of Fingerprint Recognition*: Crossing Number, lọc minutiae giả, đối sánh bằng biến đổi Hough.
- R. Važan, SourceAFIS (sourceafis.machinezoo.com); NIST NBIS (mindtct, bozorth3): baseline ngoài.
- FVC2002 (Fingerprint Verification Competition): giao thức sinh cặp genuine/impostor.
