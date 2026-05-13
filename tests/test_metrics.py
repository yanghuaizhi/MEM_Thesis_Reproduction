"""测试 UMC.utils.metric 的 ECE/AUC 计算（plan §M.1）。

依赖 UMC/utils/metric.py，本地无 GPU 也能跑（纯 numpy/sklearn）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT / "UMC"))


@pytest.fixture
def perfect_calibrated_data():
    """完美校准：y_pred = y_true mean per bin。ECE 应接近 0。"""
    import numpy as np
    rng = np.random.RandomState(42)
    y_pred = rng.rand(10_000)
    y_true = (rng.rand(10_000) < y_pred).astype(int)
    return y_pred, y_true


@pytest.fixture
def systematically_overpredicted_data():
    """系统过预测：y_pred = y_true_prob × 1.5。ECE 应较大。"""
    import numpy as np
    rng = np.random.RandomState(42)
    y_true_prob = rng.rand(10_000) * 0.3       # CTR base 0-30%
    y_pred = (y_true_prob * 1.5).clip(0, 0.99)
    y_true = (rng.rand(10_000) < y_true_prob).astype(int)
    return y_pred, y_true


def _safe_import_metric():
    try:
        from utils.metric import calibration_error
        return calibration_error
    except (ImportError, AttributeError):
        return None


def test_metric_module_importable():
    """utils.metric 应该 importable。"""
    import utils.metric                          # noqa: F401


def test_ece_perfect_calibration_low(perfect_calibrated_data):
    """完美校准数据 → ECE 应较小（< 0.05 with M=100）。"""
    calibration_error = _safe_import_metric()
    if calibration_error is None:
        pytest.skip("calibration_error not exported from utils.metric")
    y_pred, y_true = perfect_calibrated_data
    ece = calibration_error(y_pred, y_true, n_bins=100)
    assert 0 <= ece < 0.05, f"ECE for perfect calib should be < 0.05, got {ece}"


def test_ece_overpredicted_higher(systematically_overpredicted_data):
    """系统过预测 → ECE 显著 > 0。"""
    calibration_error = _safe_import_metric()
    if calibration_error is None:
        pytest.skip("calibration_error not exported")
    y_pred, y_true = systematically_overpredicted_data
    ece = calibration_error(y_pred, y_true, n_bins=100)
    assert ece > 0.05, f"systematic over-prediction should have ECE > 0.05, got {ece}"
    assert ece < 1.0


def test_ece_range_valid():
    """随机数据 ECE ∈ [0, 1]."""
    import numpy as np
    calibration_error = _safe_import_metric()
    if calibration_error is None:
        pytest.skip("calibration_error not exported")
    rng = np.random.RandomState(0)
    y_pred = rng.rand(1000)
    y_true = (rng.rand(1000) < 0.3).astype(int)
    ece = calibration_error(y_pred, y_true, n_bins=100)
    assert 0 <= ece <= 1
