"""GPU 检测与 Tier 1/2 一键硬件开关。

设计原则（plan §D.2）:
    Tier 1（完全安全，必做）: DataLoader 参数 (num_workers, pin_memory,
        persistent_workers, prefetch_factor) + eval batch_size_multiplier
    Tier 2（近无关，需 smoke test 验证）: TF32 (allow_tf32_matmul/cudnn) +
        matmul_precision + torch.compile
    Tier 3（红线，禁止动）: cudnn.benchmark, 混合精度, train batch_size

Tier 1 参数由训练代码自行读取 cfg 应用（DataLoader 必须在那里构造），
本模块只 set 进程级 Tier 2 开关；Tier 3 强制 reset 防止误改。

UMC/{pretrain,train_*}.py 在 main 开头 import:
    from reproduction.utils.gpu import setup_hardware
    setup_hardware(cfg)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def detect_gpu() -> Dict[str, Any]:
    """返回 GPU 元信息（macOS / 无 CUDA 时返回退化字典）。"""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "capability": None,
        "memory_gb": None,
        "torch_version": None,
        "cuda_version": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            idx = 0
            info["device_name"] = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            info["capability"] = f"sm_{cap[0]}{cap[1]}"
            info["memory_gb"] = round(
                torch.cuda.get_device_properties(idx).total_memory / 1024**3, 2
            )
            info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as e:
        info["error"] = str(e)
    return info


def setup_hardware(
    cfg: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """应用 Tier 1/2 硬件优化；Tier 3 强制 reset。

    Args:
        cfg: 配置字典，期望结构：
            {
              "dataloader": {num_workers, pin_memory, persistent_workers,
                             prefetch_factor},
              "eval": {batch_size_multiplier},
              "precision": {allow_tf32_matmul, allow_tf32_cudnn,
                            matmul_precision},
              "compile": {enabled},
            }
            未传或缺字段时使用安全默认值（不启 TF32，不启 compile）。
        verbose: True 时打印生效配置（便于日志审计）。

    Returns:
        实际生效配置（含 Tier 3 强制项），可写入训练 metadata。
    """
    import torch

    cfg = cfg or {}
    tier1 = dict(cfg.get("dataloader", {}))
    eval_cfg = dict(cfg.get("eval", {}))
    precision = dict(cfg.get("precision", {}))
    compile_cfg = dict(cfg.get("compile", {}))

    # Tier 2: TF32 + matmul precision
    allow_tf32_matmul = bool(precision.get("allow_tf32_matmul", False))
    allow_tf32_cudnn = bool(precision.get("allow_tf32_cudnn", False))
    matmul_precision = precision.get("matmul_precision")
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = allow_tf32_cudnn
        if matmul_precision in ("high", "medium", "highest"):
            torch.set_float32_matmul_precision(matmul_precision)
    except Exception as e:
        if verbose:
            print(f"[setup_hardware] WARN: Tier2 set failed: {e}")

    # Tier 2: torch.compile (only record flag; caller must wrap model)
    compile_enabled = bool(compile_cfg.get("enabled", False))

    # Tier 3: 强制安全默认（无视 cfg 中可能的覆盖）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    effective = {
        "tier1": tier1,
        "eval": eval_cfg,
        "tier2": {
            "allow_tf32_matmul": allow_tf32_matmul,
            "allow_tf32_cudnn": allow_tf32_cudnn,
            "matmul_precision": matmul_precision,
            "compile_enabled": compile_enabled,
        },
        "tier3_enforced": {
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "gpu": detect_gpu(),
    }

    if verbose:
        g = effective["gpu"]
        dev = g.get("device_name") or "cpu"
        mem = g.get("memory_gb")
        cap = g.get("capability")
        print(f"[setup_hardware] device={dev} mem={mem}GB cap={cap}")
        print(f"[setup_hardware] tier1 dataloader={tier1}")
        print(f"[setup_hardware] tier1 eval={eval_cfg}")
        print(
            f"[setup_hardware] tier2 tf32_matmul={allow_tf32_matmul} "
            f"tf32_cudnn={allow_tf32_cudnn} matmul_prec={matmul_precision} "
            f"compile={compile_enabled}"
        )
        print(
            "[setup_hardware] tier3 enforced: "
            "cudnn.deterministic=True cudnn.benchmark=False"
        )
    return effective


def dataloader_kwargs(
    tier1: Optional[Dict[str, Any]] = None,
    has_workers: bool = True,
) -> Dict[str, Any]:
    """从 Tier 1 配置生成 DataLoader 关键字参数，过滤 None/不兼容项。

    Args:
        tier1: setup_hardware 返回的 effective['tier1']。
        has_workers: 是否启用 num_workers>0 相关字段（influences
            persistent_workers / prefetch_factor 的合法性）。

    Returns:
        可直接 ** 展开传给 torch.utils.data.DataLoader 的字典。
    """
    tier1 = tier1 or {}
    nw = int(tier1.get("num_workers", 0))
    out: Dict[str, Any] = {
        "num_workers": nw,
        "pin_memory": bool(tier1.get("pin_memory", True)),
    }
    if has_workers and nw > 0:
        out["persistent_workers"] = bool(tier1.get("persistent_workers", True))
        pf = tier1.get("prefetch_factor")
        if pf is not None:
            out["prefetch_factor"] = int(pf)
    return out


if __name__ == "__main__":
    info = detect_gpu()
    print("GPU detect:", info)
    print()
    print("Default setup_hardware (no cfg):")
    setup_hardware(verbose=True)
    print()
    print("With Tier 1/2 cfg:")
    cfg = {
        "dataloader": {
            "num_workers": 12,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
        "eval": {"batch_size_multiplier": 4},
        "precision": {
            "allow_tf32_matmul": True,
            "allow_tf32_cudnn": True,
            "matmul_precision": "high",
        },
        "compile": {"enabled": False},
    }
    eff = setup_hardware(cfg, verbose=True)
    print()
    print("dataloader_kwargs:", dataloader_kwargs(eff["tier1"]))
