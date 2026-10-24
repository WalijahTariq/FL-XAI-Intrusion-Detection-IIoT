from __future__ import annotations

import pytest

from src import config
from tests.synthetic_data import make_synthetic_frame


@pytest.fixture()
def synthetic_csv_gz(tmp_path):
    """Write the synthetic dataset as .csv.gz (same format as the committed dataset)."""
    path = tmp_path / "ML-EdgeIIoT-dataset.csv.gz"
    make_synthetic_frame().to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    """Keep test runs from writing into the real outputs/ folder."""
    out = tmp_path / "outputs"
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    monkeypatch.setattr(config, "MODELS_DIR", out / "models")
    return out
