"""reproduction._runner — subprocess 内的 UMC 训练 wrapper。

调用方式（由 orchestrator 内部派发）:
    python -m reproduction._runner --config <path/to/run_config.json>

run_config.json schema:
    {
        "entry": "pretrain" | "train_neu" | "train_sta",
        "dataset": "aliccp" | "avazu" | "criteo",
        "config_update": { ... },           # 直接传给 UMC trial(config_update=)
        "run_dir": "<path>",                # 写 done.flag 处
        "tier": { "dataloader": {...}, "precision": {...} },  # Tier 1/2 cfg
        "log_path": "<path>"                # subprocess stdout/stderr 旁路
    }

完成后写 `<run_dir>/done.flag`（含完成时间戳）。
失败时不写 done.flag，并以非零退出码返回。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


_HERE = Path(__file__).resolve().parent              # 30_reproduction/reproduction/
_PROJECT_ROOT = _HERE.parent                          # 30_reproduction/
_UMC_DIR = _PROJECT_ROOT / "UMC"


def _setup_paths() -> None:
    """将 30_reproduction/ 和 UMC/ 加进 sys.path。"""
    for p in (str(_PROJECT_ROOT), str(_UMC_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_entry(entry: str, dataset: str):
    """根据 (entry, dataset) 返回对应的 UMC.trial 函数。"""
    _setup_paths()
    # dataset name → file suffix
    suffix_map = {"aliccp": "ali", "avazu": "avazu", "criteo": "criteo"}
    if dataset not in suffix_map:
        raise ValueError(f"Unknown dataset: {dataset}")
    suffix = suffix_map[dataset]

    if entry == "pretrain":
        from pretrain import trial as _trial
        return _trial
    elif entry == "train_neu":
        mod = __import__(f"train_neu_{suffix}")
        return mod.trial
    elif entry == "train_sta":
        mod = __import__(f"train_sta_{suffix}")
        return mod.trial
    elif entry == "v9_inference":
        # TODO(task #11 / #2): 从 _legacy_runs/run_v9_error_analysis.py 抽 inference 段
        # 关键约束（plan §A.5.2 第 1 条审计项）:
        #   - 必须用未校准 backbone 输出（PackedDeepFM forward，不经 calibrator）
        #   - 保存 NPZ 字段: y_pred_uncalib, y_pred_calib, y_true, u, sigma_sq
        raise NotImplementedError(
            "v9_inference entry not yet implemented in _runner.py. "
            "See _legacy_runs/run_v9_error_analysis.py L120-200 for inference logic."
        )
    else:
        raise ValueError(f"Unknown entry: {entry}")


def _apply_tier(tier: dict) -> dict:
    """调用 reproduction.utils.gpu.setup_hardware 应用 Tier 1/2。"""
    _setup_paths()
    from reproduction.utils.gpu import setup_hardware
    return setup_hardware(tier, verbose=True)


def main():
    ap = argparse.ArgumentParser(prog="reproduction._runner")
    ap.add_argument("--config", required=True, help="Path to run_config.json")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        run = json.load(f)

    entry: str = run["entry"]
    dataset: str = run["dataset"]
    config_update: dict = run["config_update"]
    run_dir = Path(run["run_dir"])
    tier = run.get("tier", {})

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    print(f"[runner] entry={entry} dataset={dataset} run_dir={run_dir}")
    print(f"[runner] config_update_keys={sorted(config_update.keys())}")

    # Tier 1/2 优化（gpu.py 强制 Tier 3 红线）
    try:
        _apply_tier(tier)
    except Exception as e:
        print(f"[runner] WARN: setup_hardware failed: {e}")

    # 解析 trial 函数并执行
    try:
        trial_fn = _resolve_entry(entry, dataset)
        trial_fn(config_update=config_update)
    except SystemExit as e:
        # UMC trial 内部如果 sys.exit，我们记录但不写 done.flag
        if int(e.code or 0) != 0:
            _write_status(run_dir, "exit_nonzero", started_at, code=int(e.code or 0))
            raise
    except Exception as e:
        _write_status(run_dir, "error", started_at, error=str(e), traceback=traceback.format_exc())
        raise

    # 成功 → 写 done.flag
    _write_status(run_dir, "done", started_at)


def _write_status(run_dir: Path, status: str, started_at: float, **extra) -> None:
    payload = {
        "status": status,
        "started_at": started_at,
        "ended_at": time.time(),
        "elapsed_sec": round(time.time() - started_at, 2),
        **extra,
    }
    if status == "done":
        (run_dir / "done.flag").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        (run_dir / "error.flag").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
