"""
UMC/_paths.py — 路径与依赖解析（30_reproduction 复现项目专用）

设计目的:
    取代原 baiyimeng/UMC 代码中硬编码的 /root/shared-nvme/PAPER/ 路径，
    通过环境变量 + 与 30_reproduction 目录对齐的 fallback 实现自动适配。

环境变量（按优先级使用，未设置时用 fallback）:
    MEM_DATA_ROOT              数据集根目录（含 aliccp/avazu/criteo 子目录）
    MEM_CKPT_ROOT              checkpoint 与训练产物根目录
    MEM_TORCH_UNCERTAINTY_SRC  torch-uncertainty 库源码路径（PackedEnsemble 依赖）

Fallback（与 30_reproduction/ 目录布局对齐）:
    DATA_ROOT              = <30_reproduction>/data/processed
    CKPT_ROOT              = <30_reproduction>/experiments
    TORCH_UNCERTAINTY_SRC  = <30_reproduction>/../10_research_archive/_archive/torch-uncertainty/src
                             （首选，本地开发用）
                          或 <30_reproduction>/UMC/../_archive/torch-uncertainty/src
                             （如有内嵌副本）

注意:
    本模块**仅供 UMC/ 内部使用**，不要在 reproduction/ 内引用。
    reproduction/ 的路径配置走 configs/paths.yaml + reproduction/utils/。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent           # 30_reproduction/UMC/
_PROJECT_ROOT = _HERE.parent                       # 30_reproduction/
_REPO_ROOT = _PROJECT_ROOT.parent                  # /Users/y/Research_MEM/ (or remote equivalent)


# ============================================================================
# Public constants
# ============================================================================

DATA_ROOT: str = os.environ.get(
    "MEM_DATA_ROOT",
    str(_PROJECT_ROOT / "data" / "processed"),
)

CKPT_ROOT: str = os.environ.get(
    "MEM_CKPT_ROOT",
    str(_PROJECT_ROOT / "experiments"),
)

# torch-uncertainty 解析优先级:
#   1. env var MEM_TORCH_UNCERTAINTY_SRC（如显式指定）
#   2. <repo_root>/10_research_archive/_archive/torch-uncertainty/src（本地开发）
#   3. <30_reproduction>/_archive/torch-uncertainty/src（远程容器内嵌副本）
#   4. None（如果都没有，靠 pip install torch-uncertainty fallback）
_TORCH_UNCERTAINTY_CANDIDATES = [
    os.environ.get("MEM_TORCH_UNCERTAINTY_SRC", ""),
    str(_REPO_ROOT / "10_research_archive" / "_archive" / "torch-uncertainty" / "src"),
    str(_PROJECT_ROOT / "_archive" / "torch-uncertainty" / "src"),
]

TORCH_UNCERTAINTY_SRC: str | None = next(
    (p for p in _TORCH_UNCERTAINTY_CANDIDATES if p and os.path.isdir(p)),
    None,
)


# ============================================================================
# Public helpers
# ============================================================================

def setup_torch_uncertainty(verbose: bool = False) -> bool:
    """将 torch-uncertainty 源码路径添加到 sys.path。

    Returns:
        True if 已添加到 sys.path 或已 importable；False 如均未找到。
    """
    if TORCH_UNCERTAINTY_SRC and TORCH_UNCERTAINTY_SRC not in sys.path:
        sys.path.append(TORCH_UNCERTAINTY_SRC)
        if verbose:
            print(f"[_paths] torch-uncertainty: {TORCH_UNCERTAINTY_SRC}")
        return True
    # 已在 sys.path 或路径未找到——尝试 import 检测
    try:
        import torch_uncertainty  # noqa: F401
        return True
    except ImportError:
        if verbose:
            print(
                f"[_paths] WARNING: torch-uncertainty not found. "
                f"Tried: {_TORCH_UNCERTAINTY_CANDIDATES}. "
                f"Some PackedEnsemble features may fail. "
                f"Set MEM_TORCH_UNCERTAINTY_SRC env var or pip install torch-uncertainty."
            )
        return False


def uncertainty_bin_path(dataset_name: str) -> str:
    """返回 phase4_uncertainty_bins_<dataset>.csv 的标准路径。"""
    return str(Path(CKPT_ROOT) / f"phase4_uncertainty_bins_{dataset_name}.csv")


def dataset_path(dataset_name: str) -> str:
    """返回某数据集的根目录（包含 train/val/test parquet）。"""
    return str(Path(DATA_ROOT) / dataset_name)


def ensure_dirs() -> None:
    """确保 DATA_ROOT 与 CKPT_ROOT 存在（首次运行时创建）。"""
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(CKPT_ROOT).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Debug entry point
# ============================================================================

if __name__ == "__main__":
    print(f"DATA_ROOT             = {DATA_ROOT}")
    print(f"CKPT_ROOT             = {CKPT_ROOT}")
    print(f"TORCH_UNCERTAINTY_SRC = {TORCH_UNCERTAINTY_SRC}")
    print(f"")
    print(f"Existence check:")
    print(f"  DATA_ROOT exists:  {os.path.isdir(DATA_ROOT)}")
    print(f"  CKPT_ROOT exists:  {os.path.isdir(CKPT_ROOT)}")
    print(f"  TORCH_UNC exists:  {TORCH_UNCERTAINTY_SRC is not None}")
    print(f"")
    print(f"setup_torch_uncertainty(verbose=True):")
    ok = setup_torch_uncertainty(verbose=True)
    print(f"  result: {ok}")
