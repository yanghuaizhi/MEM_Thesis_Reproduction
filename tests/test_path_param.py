"""测试 UMC/_paths.py 解析（smoke：路径参数化生效）。

plan §M.1 验证：路径参数化前后输出 1e-6 容差（实际跑需 GPU+数据，本地仅检查
解析逻辑 + 模块导入不抛错）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT / "UMC"))


def test_paths_module_imports():
    import _paths

    assert hasattr(_paths, "DATA_ROOT")
    assert hasattr(_paths, "CKPT_ROOT")
    assert hasattr(_paths, "setup_torch_uncertainty")


def test_paths_resolve_to_30_reproduction():
    """无 env var 时应 fallback 到 30_reproduction/data/processed."""
    # 清掉 env var 测试 fallback
    saved = {k: os.environ.pop(k, None) for k in ("MEM_DATA_ROOT", "MEM_CKPT_ROOT")}
    try:
        # 重新 import _paths（绕过缓存）
        import importlib
        import _paths
        importlib.reload(_paths)
        assert _paths.DATA_ROOT.endswith("30_reproduction/data/processed")
        assert _paths.CKPT_ROOT.endswith("30_reproduction/experiments")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_paths_respect_env_var(tmp_path, monkeypatch):
    """MEM_DATA_ROOT env var 应覆盖 fallback。"""
    monkeypatch.setenv("MEM_DATA_ROOT", str(tmp_path))
    import importlib
    import _paths
    importlib.reload(_paths)
    assert _paths.DATA_ROOT == str(tmp_path)


def test_no_legacy_hardcoded_paths():
    """UMC/*.py 不应残留 /root/shared-nvme 或 /data/baiyimeng 硬编码（_paths.py docstring 中除外）。"""
    umc_dir = _PROJECT / "UMC"
    issues = []
    for py in umc_dir.glob("*.py"):
        if py.name == "_paths.py":
            continue
        text = py.read_text(encoding="utf-8")
        for needle in ("/root/shared-nvme/", "/data/baiyimeng/"):
            if needle in text:
                issues.append(f"{py.name}: contains '{needle}'")
    assert not issues, f"Legacy hardcoded paths found: {issues}"
