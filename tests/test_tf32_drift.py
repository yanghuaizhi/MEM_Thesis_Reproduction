"""TF32 数值差异验证（plan §D.2 验收）。

要求: AliCCP × seed=1024 × UMC × 2 epoch 下，TF32 ON/OFF 的 |ΔECE| < 1e-5。

**需 GPU + 数据**，本地跑会 skip。设计为远程容器内 pytest 触发的 smoke test。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT))


@pytest.mark.skipif(
    os.environ.get("MEM_SMOKE_TEST_TF32", "0") != "1",
    reason="set MEM_SMOKE_TEST_TF32=1 to enable (requires GPU + data)",
)
def test_tf32_drift_smoke():
    """端到端 smoke：跑 2 epoch AliCCP UMC，TF32 ON vs OFF。

    实际执行步骤（待数据 ready 后填充）:
        1. setup_hardware({"precision": {"allow_tf32_matmul": False, ...}})
        2. orchestrator 跑 main aliccp umc seed=1024 max-epochs=2 → ECE_OFF
        3. setup_hardware({"precision": {"allow_tf32_matmul": True, ...}})
        4. 重跑 → ECE_ON
        5. assert abs(ECE_OFF - ECE_ON) < 1e-5
    """
    pytest.skip("TF32 drift smoke test requires GPU + preprocessed data")


def test_setup_hardware_with_tf32_off_no_crash():
    """轻量 smoke: setup_hardware with TF32 OFF 不崩溃（不需要 GPU）。"""
    from reproduction.utils.gpu import setup_hardware

    eff = setup_hardware(
        cfg={
            "precision": {
                "allow_tf32_matmul": False,
                "allow_tf32_cudnn": False,
                "matmul_precision": "highest",
            }
        },
        verbose=False,
    )
    assert eff["tier2"]["allow_tf32_matmul"] is False
    assert eff["tier3_enforced"]["cudnn_benchmark"] is False


def test_setup_hardware_with_tf32_on_no_crash():
    """轻量 smoke: setup_hardware with TF32 ON 不崩溃。"""
    from reproduction.utils.gpu import setup_hardware

    eff = setup_hardware(
        cfg={
            "precision": {
                "allow_tf32_matmul": True,
                "allow_tf32_cudnn": True,
                "matmul_precision": "high",
            }
        },
        verbose=False,
    )
    assert eff["tier2"]["allow_tf32_matmul"] is True
    assert eff["tier2"]["matmul_precision"] == "high"
