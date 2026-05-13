"""测试 orchestrator 的 plan 生成 + filter 逻辑。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def configs():
    from reproduction.orchestrator import _load_configs

    return _load_configs()


def test_pretrain_plan_has_3_runs(configs):
    """FIX-3: pretrain 只跑 3 个（每数据集 seed=1024 固定）。

    backbone checkpoint 文件名编码 seed，calib 永远加载 seed=1024 的 backbone，
    所以 seed=2024/3024 的 backbone 训完即弃，必须省。
    """
    from reproduction.orchestrator import _build_pretrain_plan

    runs = _build_pretrain_plan(configs)
    assert len(runs) == 3                    # 3 datasets × 1 fixed seed
    seeds = {r["seed"] for r in runs}
    assert seeds == {1024}                   # 固定 1024
    datasets = sorted({r["dataset"] for r in runs})
    assert datasets == ["aliccp", "avazu", "criteo"]


def test_main_plan_has_99_runs(configs):
    from reproduction.orchestrator import _build_main_plan

    runs = _build_main_plan(configs, None, None, None)
    assert len(runs) == 99                   # 11 × 3 × 3


def test_main_plan_filter_by_dataset(configs):
    from reproduction.orchestrator import _build_main_plan

    runs = _build_main_plan(configs, "aliccp", None, None)
    assert len(runs) == 33                   # 11 methods × 3 seeds
    assert all(r["dataset"] == "aliccp" for r in runs)


def test_main_plan_filter_by_method(configs):
    from reproduction.orchestrator import _build_main_plan

    runs = _build_main_plan(configs, None, "uamcm", None)
    assert len(runs) == 9                    # 3 datasets × 3 seeds
    assert all(r["method"] == "uamcm" for r in runs)


def test_main_plan_filter_by_method_type(configs):
    from reproduction.orchestrator import _build_main_plan

    runs = _build_main_plan(configs, None, None, "statistical")
    assert len(runs) == 27                   # 3 stat methods × 3 datasets × 3 seeds
    methods = {r["method"] for r in runs}
    assert methods == {"platt", "ir", "hb"}


def test_v10_plan_has_27_runs(configs):
    from reproduction.orchestrator import _build_v10_plan

    runs = _build_v10_plan(configs)
    assert len(runs) == 27                   # 3 u_modes × 3 datasets × 3 seeds
    u_modes_in_method = {r["method"].split("_umode_")[1] for r in runs}
    assert u_modes_in_method == {"pe", "shuffled", "logit"}


def test_config_update_critical_fields(configs):
    """关键字段（plan §C）经过 orchestrator 后必须正确传给 UMC。"""
    from reproduction.orchestrator import _build_main_plan

    runs = _build_main_plan(configs, "aliccp", "uamcm", None)
    cu = runs[0]["config_update"]            # 任一 seed 都行
    assert cu["num_estimators"] == 16
    assert cu["ece_M"] == 100
    assert cu["field_index"] == 0            # aliccp
    assert cu["seed"] == 1024                # backbone seed 固定
    assert cu["calib_seed"] in (1024, 2024, 3024)
    assert cu["batch_size_calib"] == 65536   # AliCCP non-OOM-safe
    # FIX-4: integral_dim 字段已删，UAMCM 不应传入
    assert "integral_dim" not in cu
    # FIX-5: uncertainty_bin_save_path 必须注入
    assert "uncertainty_bin_save_path" in cu


def test_avazu_uamcm_uses_16k_batch(configs):
    """Avazu UAMCM 必须用 16K calib batch（plan §B 第 5 条）。"""
    from reproduction.orchestrator import _build_main_plan

    runs = _build_main_plan(configs, "avazu", "uamcm", None)
    for r in runs:
        assert r["config_update"]["batch_size_calib"] == 16384
