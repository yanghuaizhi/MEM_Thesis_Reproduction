"""测试 reproduction.utils 的 4 个工具模块。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))


def test_setup_seed_idempotent():
    from reproduction.utils import setup_seed
    import numpy as np

    setup_seed(1024)
    a = np.random.rand(5)
    setup_seed(1024)
    b = np.random.rand(5)
    assert (a == b).all()


def test_derive_seed_deterministic():
    from reproduction.utils import derive_seed

    a = derive_seed(1024, "aliccp", "umc")
    b = derive_seed(1024, "aliccp", "umc")
    c = derive_seed(1024, "aliccp", "uamcm")
    assert a == b
    assert a != c
    assert 0 <= a < 2**31 - 1


def test_jsonl_logger_writes_and_reads():
    from reproduction.utils import JsonlLogger, jsonl_iter

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        path = tmp.name
    try:
        with JsonlLogger(path, also_stdout=False) as L:
            L.log("epoch_end", epoch=3, ece=0.073)
            L.log("final", ece=0.071, auc=0.72)
        records = list(jsonl_iter(path))
        assert len(records) == 2
        assert records[0]["event"] == "epoch_end"
        assert abs(records[1]["ece"] - 0.071) < 1e-9
    finally:
        Path(path).unlink(missing_ok=True)


def test_write_status_atomic():
    from reproduction.utils import write_status, read_status, update_phase

    # write_status 应支持任意 status path
    from reproduction.utils.status import read_status

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        write_status({"phase": "test", "x": 1}, path=path)
        s = read_status(path)
        assert s["phase"] == "test"
        assert s["x"] == 1
        assert "timestamp" in s
        update_phase("test2", path=path, y=2)
        s = read_status(path)
        assert s["phase"] == "test2"
        assert s["x"] == 1               # 保留旧字段
        assert s["y"] == 2
    finally:
        Path(path).unlink(missing_ok=True)


def test_setup_hardware_enforces_tier3():
    """Tier 3 红线必须始终 enforce。"""
    from reproduction.utils import setup_hardware

    eff = setup_hardware(
        cfg={
            "precision": {"allow_tf32_matmul": True},
            "tier3_locked": {"cudnn_benchmark": True},   # 即使用户传 True
        },
        verbose=False,
    )
    assert eff["tier3_enforced"]["cudnn_benchmark"] is False
    assert eff["tier3_enforced"]["cudnn_deterministic"] is True


def test_dataloader_kwargs_filtering():
    from reproduction.utils.gpu import dataloader_kwargs

    out = dataloader_kwargs({"num_workers": 0, "pin_memory": True})
    assert out["num_workers"] == 0
    assert "persistent_workers" not in out          # num_workers=0 时禁用

    out = dataloader_kwargs({"num_workers": 4, "pin_memory": True, "prefetch_factor": 2})
    assert out["num_workers"] == 4
    assert out["persistent_workers"] is True
    assert out["prefetch_factor"] == 2
