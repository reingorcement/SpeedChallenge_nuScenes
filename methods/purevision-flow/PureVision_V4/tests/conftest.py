from pathlib import Path
import tempfile

import pytest
import torch

from purevision_v4.io import ROOT, calibration, load_config, read_json


@pytest.fixture
def cal():
    return calibration(read_json(ROOT / "tests/calibration_legacy.json"))


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def work():
    base = ROOT / "verification/test_tmp"
    base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        yield Path(directory)


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)
