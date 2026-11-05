import io

import numpy as np
from PIL import Image

from fpsys.io import load_image, to_gray


def _jpeg_with_orientation(orientation: int) -> bytes:
    arr = np.zeros((40, 80, 3), np.uint8)
    arr[:, :40] = (255, 0, 0)  # nửa trái màu đỏ (RGB)
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes(), quality=95)
    return buf.getvalue()


def test_exif_rotation_applied_bytes_and_path(tmp_path):
    data = _jpeg_with_orientation(6)  # xoay 90° khi hiển thị
    img = load_image(data)
    assert img.shape[:2] == (80, 40)
    p = tmp_path / "x.jpg"
    p.write_bytes(data)
    assert np.array_equal(load_image(str(p)), img)


def test_bgr_order():
    img = load_image(_jpeg_with_orientation(1))
    assert img.shape == (40, 80, 3)
    assert img[20, 10, 2] > 200 and img[20, 10, 0] < 60  # đỏ → kênh R ở chỉ số 2


def test_to_gray_channels():
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 1] = 200
    assert (to_gray(img, "green") == 200).all()
    assert to_gray(img, "gray").shape == (4, 4)
