"""测试 reproduction/configs/ 19 个 YAML 文件加载与一致性。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


_PROJECT = Path(__file__).parent.parent
_CFG = _PROJECT / "reproduction" / "configs"


def test_all_yamls_parse():
    files = sorted(_CFG.rglob("*.yaml"))
    assert len(files) == 19, f"expected 19 YAMLs, got {len(files)}"
    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "name" in data, f"{f.name} missing 'name' field"


def test_main_99_consistency():
    """main_99.methods/datasets 与 disk 完全一致。"""
    main = yaml.safe_load((_CFG / "experiments" / "main_99.yaml").read_text())
    disk_methods = {p.stem for p in (_CFG / "methods").glob("*.yaml")}
    disk_datasets = {p.stem for p in (_CFG / "datasets").glob("*.yaml")}
    assert set(main["methods"]) == disk_methods
    assert set(main["datasets"]) == disk_datasets
    assert len(main["methods"]) == 11
    assert len(main["datasets"]) == 3
    assert main["seeds"] == [1024, 2024, 3024]


def test_critical_constants():
    """plan §C 关键不可变常量校验。"""
    main = yaml.safe_load((_CFG / "experiments" / "main_99.yaml").read_text())
    assert main["backbone"]["num_estimators"] == 16, "M=16 不可改 (plan §C.1)"
    assert main["backbone"]["pretrain_seed"] == 1024, "pretrain seed 固定 1024"
    assert main["evaluation"]["ece_bins"] == 100, "ECE M=100 不可改 (plan §C.2)"
    assert main["evaluation"]["std_ddof"] == 1, "ddof=1 Bessel 校正 (plan §B 第 6 条)"


def test_field_index_per_dataset():
    """field_index 严守 aliccp=0 / avazu=2 / criteo=23。"""
    expected = {"aliccp": 0, "avazu": 2, "criteo": 23}
    for ds, idx in expected.items():
        cfg = yaml.safe_load((_CFG / "datasets" / f"{ds}.yaml").read_text())
        assert cfg["field_index"] == idx, f"{ds} field_index={cfg['field_index']} != {idx}"


def test_avazu_oom_safe_batch():
    """Avazu calib batch=16384 (plan §B 第 5 条避坑)。"""
    cfg = yaml.safe_load((_CFG / "datasets" / "avazu.yaml").read_text())
    assert cfg["batch_size"]["pretrain"] == 16384
    assert cfg["batch_size"]["calib"] == 16384


def test_method_entry_types():
    """statistical 走 train_sta，其他走 train_neu。"""
    for f in (_CFG / "methods").glob("*.yaml"):
        cfg = yaml.safe_load(f.read_text())
        if cfg["type"] == "statistical":
            assert cfg["entry"] == "train_sta"
            assert cfg["uses_u"] is False
        else:
            assert cfg["entry"] == "train_neu"


def test_paper_core_uses_u():
    """所有 paper_core 方法（umc/umc_wor/uamcm/uamcm_wor）uses_u=True。"""
    for name in ("umc", "umc_wor", "uamcm", "uamcm_wor"):
        cfg = yaml.safe_load((_CFG / "methods" / f"{name}.yaml").read_text())
        assert cfg["type"] == "paper_core"
        assert cfg["uses_u"] is True


def test_uamcm_no_invalid_integral_dim():
    """FIX-4: integral_dim 应已从 YAML 移除（UAMCM 构造函数无此参数）。"""
    cfg = yaml.safe_load((_CFG / "methods" / "uamcm.yaml").read_text())
    assert "integral_dim" not in cfg.get("hyperparameters", {})
    cfg_wor = yaml.safe_load((_CFG / "methods" / "uamcm_wor.yaml").read_text())
    assert "integral_dim" not in cfg_wor.get("hyperparameters", {})


def test_hardware_tier3_locked():
    cfg = yaml.safe_load((_CFG / "hardware" / "rtx5090.yaml").read_text())
    assert cfg["tier3_locked"]["cudnn_benchmark"] is False
    assert cfg["tier3_locked"]["cudnn_deterministic"] is True
    assert cfg["tier3_locked"]["mixed_precision"] is False
