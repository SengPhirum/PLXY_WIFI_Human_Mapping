import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wifi_mapping.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config()
    # Small windows so tests run on short streams.
    c.preprocess.window_size = 50
    c.preprocess.window_step = 25
    c.preprocess.n_pca_components = 0
    return c
