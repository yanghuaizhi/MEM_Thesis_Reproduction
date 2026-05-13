"""测试 diff_with_paper 的判定逻辑（mock data 驱动）。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_main_records():
    """构造一组覆盖三数据集 + 关键方法的 records。"""
    out = []
    # AliCCP: UAMCM 比 UMC 改善 13.7%（与论文一致）
    for seed, umc_ece, uamcm_ece in [(1024, 0.085, 0.075), (2024, 0.090, 0.073), (3024, 0.083, 0.075)]:
        out.append({"dataset": "aliccp", "method": "umc", "seed": seed, "ece": umc_ece, "auc": 0.71, "logloss": 0.45})
        out.append({"dataset": "aliccp", "method": "uamcm", "seed": seed, "ece": uamcm_ece, "auc": 0.72, "logloss": 0.44})
    # Criteo: UAMCM 改善 46%
    for seed in (1024, 2024, 3024):
        out.append({"dataset": "criteo", "method": "umc", "seed": seed, "ece": 0.048, "auc": 0.78, "logloss": 0.42})
        out.append({"dataset": "criteo", "method": "uamcm", "seed": seed, "ece": 0.026, "auc": 0.79, "logloss": 0.41})
        out.append({"dataset": "criteo", "method": "ir", "seed": seed, "ece": 0.025, "auc": 0.77, "logloss": 0.43})
    # Avazu: UAMCM 改善 17.6%
    for seed in (1024, 2024, 3024):
        out.append({"dataset": "avazu", "method": "umc", "seed": seed, "ece": 0.13, "auc": 0.75, "logloss": 0.40})
        out.append({"dataset": "avazu", "method": "uamcm", "seed": seed, "ece": 0.107, "auc": 0.76, "logloss": 0.39})
    return out


@pytest.fixture
def mock_v10_records():
    """v10 records: AliCCP+Criteo 应支持 shuffled 恶化；Avazu 应在 σ 内。

    注意 Avazu 用真实的多 seed 方差（σ>0），否则判定逻辑会因 σ=0 而 false-opposes。
    """
    out = []
    # AliCCP: shuffled 显著恶化（~73%）
    for seed, pe_e, sh_e, lg_e in [
        (1024, 0.075, 0.130, 0.080),
        (2024, 0.073, 0.128, 0.082),
        (3024, 0.077, 0.132, 0.080),
    ]:
        out.append({"dataset": "aliccp", "method": "uamcm", "u_mode": "pe", "seed": seed, "ece": pe_e})
        out.append({"dataset": "aliccp", "method": "uamcm", "u_mode": "shuffled", "seed": seed, "ece": sh_e})
        out.append({"dataset": "aliccp", "method": "uamcm", "u_mode": "logit", "seed": seed, "ece": lg_e})
    # Criteo: shuffled 显著恶化（~70%）
    for seed, pe_e, sh_e, lg_e in [
        (1024, 0.026, 0.044, 0.030),
        (2024, 0.025, 0.043, 0.029),
        (3024, 0.027, 0.045, 0.031),
    ]:
        out.append({"dataset": "criteo", "method": "uamcm", "u_mode": "pe", "seed": seed, "ece": pe_e})
        out.append({"dataset": "criteo", "method": "uamcm", "u_mode": "shuffled", "seed": seed, "ece": sh_e})
        out.append({"dataset": "criteo", "method": "uamcm", "u_mode": "logit", "seed": seed, "ece": lg_e})
    # Avazu: shuffled 落在 σ 内（高 CV ~ 40%，shuffled 变化 < σ）
    for seed, pe_e, sh_e, lg_e in [
        (1024, 0.10, 0.105, 0.108),
        (2024, 0.13, 0.125, 0.130),
        (3024, 0.16, 0.155, 0.155),
    ]:
        out.append({"dataset": "avazu", "method": "uamcm", "u_mode": "pe", "seed": seed, "ece": pe_e})
        out.append({"dataset": "avazu", "method": "uamcm", "u_mode": "shuffled", "seed": seed, "ece": sh_e})
        out.append({"dataset": "avazu", "method": "uamcm", "u_mode": "logit", "seed": seed, "ece": lg_e})
    return out


def test_aggregate_mean_std_ddof1(mock_main_records):
    from reproduction.analysis.diff_with_paper import aggregate_mean_std

    agg = aggregate_mean_std(mock_main_records, "ece", ddof=1)
    aliccp_umc = agg[("aliccp", "umc")]
    assert aliccp_umc["n"] == 3
    # mean = (0.085+0.090+0.083)/3 ≈ 0.086
    assert abs(aliccp_umc["mean"] - 0.086) < 1e-3


def test_p2_aliccp_supports(mock_main_records):
    """AliCCP UAMCM 比 UMC 改善 ~13% → supports P2 (threshold 5%)."""
    from reproduction.analysis.diff_with_paper import check_p2_p3_p4

    v = check_p2_p3_p4(mock_main_records)
    assert v["P2"].state == "supports"
    assert v["P2"].reproduction_value["improvement_pct"] > 5


def test_p3_criteo_supports(mock_main_records):
    """Criteo UAMCM 改善 ~46% → supports P3 (threshold 25%)."""
    from reproduction.analysis.diff_with_paper import check_p2_p3_p4

    v = check_p2_p3_p4(mock_main_records)
    assert v["P3"].state == "supports"
    assert v["P3"].reproduction_value["improvement_pct"] > 25


def test_p5_aliccp_shuffled_supports(mock_v10_records):
    """AliCCP shuffled +73% → supports P5 (threshold 30%)."""
    from reproduction.analysis.diff_with_paper import check_p5_shuffled

    v = check_p5_shuffled(mock_v10_records)
    assert v["P5_aliccp"].state == "supports"


def test_p5_avazu_in_sigma_supports(mock_v10_records):
    """Avazu shuffled 在 σ 内 → supports P4 (核心反例)."""
    from reproduction.analysis.diff_with_paper import check_p5_shuffled

    v = check_p5_shuffled(mock_v10_records)
    # 注意：判定与具体 std 相关
    assert v["P5_avazu"].state in ("supports", "neutral")
    # 验证 worsening_pct 接近 0
    assert abs(v["P5_avazu"].reproduction_value["worsening_pct"]) < 10


def test_s1_criteo_statistical(mock_main_records):
    """Criteo IR 进 top-3 → S1 supports."""
    from reproduction.analysis.diff_with_paper import check_decision_framework

    v = check_decision_framework(mock_main_records, [])
    # mock 数据中 criteo IR ECE=0.025 是最小，进 top-3
    assert v["S1"].state == "supports"


def test_no_data_returns_no_data_verdict():
    """空 records 时所有判定返回 no_data 而不崩溃。"""
    from reproduction.analysis.diff_with_paper import (
        check_p2_p3_p4, check_p5_shuffled, check_decision_framework,
    )

    for v in check_p2_p3_p4([]).values():
        assert v.state == "no_data"
    for v in check_p5_shuffled([]).values():
        assert v.state == "no_data"
    decisions = check_decision_framework([], [])
    assert all(v.state == "no_data" for v in decisions.values())


def test_compute_pcoc():
    from reproduction.analysis.diff_with_paper import compute_pcoc

    pcoc = compute_pcoc([0.5, 0.6, 0.7], [0.4, 0.5, 0.6])
    assert abs(pcoc - 0.6 / 0.5) < 1e-9


def test_monotonic_detection():
    from reproduction.analysis.diff_with_paper import _check_monotonic

    assert _check_monotonic([1.5, 1.2, 1.0, 0.8]) == "decreasing"
    assert _check_monotonic([0.6, 0.8, 1.0, 1.2]) == "increasing"
    assert _check_monotonic([0.6, 1.0, 0.8, 1.2]) == "neither"
