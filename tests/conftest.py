import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpsys.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()
