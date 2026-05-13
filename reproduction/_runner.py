"""reproduction._runner — subprocess 内的 UMC 训练 wrapper。

调用方式（由 orchestrator 内部派发）:
    python -m reproduction._runner --config <path/to/run_config.json>

run_config.json schema:
    {
        "entry": "pretrain" | "train_neu" | "train_sta",
        "dataset": "aliccp" | "avazu" | "criteo",
        "config_update": { ... },           # 直接传给 UMC trial(config_update=)
        "run_dir": "<path>",                # 写 done.flag 处
        "tier": { ... }                     # Tier 1/2 cfg
    }

完成后:
    <run_dir>/done.flag        含完成时间戳
    <run_dir>/metrics.jsonl    从 UMC stdout 提取的指标（B1）

失败时不写 done.flag，并以非零退出码返回。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


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
    else:
        raise ValueError(f"Unknown entry: {entry}")


def _apply_tier(tier: dict) -> dict:
    _setup_paths()
    from reproduction.utils.gpu import setup_hardware
    return setup_hardware(tier, verbose=True)


# ============================================================================
# Metrics extraction (B1: train.log → metrics.jsonl)
# ============================================================================

class CapturingTee:
    """Mirror stdout to real fd + in-memory buffer (for metrics extraction)."""

    def __init__(self, real_stdout):
        self.real = real_stdout
        self._lines: List[str] = []
        self._partial = ""

    def write(self, s: str) -> int:
        self.real.write(s)
        self.real.flush()
        # 按行 buffer
        self._partial += s
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._lines.append(line)
        return len(s)

    def flush(self) -> None:
        self.real.flush()

    def get_lines(self) -> List[str]:
        if self._partial:
            self._lines.append(self._partial)
            self._partial = ""
        return list(self._lines)


# UMC `evaluate()` 输出格式: "test_auc = 0.706667, test_logloss = 0.452100, test_ece = 0.073210, ..."
_METRIC_PATTERN = re.compile(r"test_(\w+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_TAG_PATTERN = re.compile(r"^metrics_tag=(\S+)")
_SHUFFLED_CORR_PATTERN = re.compile(r"shuffled_u_pearson_corr=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def extract_metrics(lines: List[str]) -> List[Dict[str, Any]]:
    """从 UMC train.log 文本提取 metrics records。

    UMC 输出格式（每个 metrics_tag 段后 0-5 行内有 test_xxx = N.NNNNN 形式的指标）:
        metrics_tag=calibrated
        test_auc = 0.7067, test_gauc = 0.6500, test_logloss = 0.4521, ... test_ece = 0.0732, ...

    返回每个 tag 对应的 record dict。
    """
    records: List[Dict[str, Any]] = []
    n = len(lines)
    for i, line in enumerate(lines):
        m = _TAG_PATTERN.match(line.strip())
        if not m:
            continue
        tag = m.group(1)
        # 搜后续最多 5 行找 test_xxx 指标
        rec: Dict[str, Any] = {"event": "metrics", "metrics_tag": tag}
        for j in range(i + 1, min(i + 6, n)):
            for k, v in _METRIC_PATTERN.findall(lines[j]):
                try:
                    rec[k] = float(v)
                except ValueError:
                    pass
            if any(key in rec for key in ("auc", "ece", "logloss")):
                break
        if any(k in rec for k in ("auc", "ece", "logloss")):
            records.append(rec)

    # 单独提取 shuffled_u_pearson_corr（B6 audit）
    for line in lines:
        m = _SHUFFLED_CORR_PATTERN.search(line)
        if m:
            try:
                records.append({
                    "event": "shuffled_u_audit",
                    "shuffled_u_pearson_corr": float(m.group(1)),
                })
            except ValueError:
                pass
            break

    return records


def write_metrics_jsonl(records: List[Dict[str, Any]], run_dir: Path) -> Path:
    """写 metrics.jsonl 到 run_dir。同时挑出关键的 'final' record（calibrated tag 优先）。"""
    out = run_dir / "metrics.jsonl"
    # 标记 final = calibrated tag (loss-best 主结果，plan §B 第 7 条避坑)
    final_rec = None
    for r in records:
        if r.get("metrics_tag") == "calibrated":
            final_rec = {**r, "event": "final"}
    if final_rec is None:
        # 退而求其次：用任何含 ece 的记录
        for r in records:
            if "ece" in r:
                final_rec = {**r, "event": "final"}
                break

    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if final_rec is not None:
            f.write(json.dumps(final_rec, ensure_ascii=False) + "\n")
    return out


# ============================================================================
# Main entry
# ============================================================================

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

    try:
        _apply_tier(tier)
    except Exception as e:
        print(f"[runner] WARN: setup_hardware failed: {e}")

    # B1: 安装 stdout tee 捕获 UMC 输出用于 metrics 提取
    tee = CapturingTee(sys.stdout)
    sys.stdout = tee

    try:
        trial_fn = _resolve_entry(entry, dataset)
        trial_fn(config_update=config_update)
    except SystemExit as e:
        sys.stdout = tee.real
        if int(e.code or 0) != 0:
            _write_status(run_dir, "exit_nonzero", started_at, code=int(e.code or 0))
            _flush_metrics(tee, run_dir)
            raise
    except Exception as e:
        sys.stdout = tee.real
        _write_status(run_dir, "error", started_at,
                      error=str(e), traceback=traceback.format_exc())
        _flush_metrics(tee, run_dir)
        raise
    finally:
        sys.stdout = tee.real

    # 成功 → 写 metrics.jsonl + done.flag
    _flush_metrics(tee, run_dir)
    _write_status(run_dir, "done", started_at)


def _flush_metrics(tee: CapturingTee, run_dir: Path) -> None:
    try:
        records = extract_metrics(tee.get_lines())
        if records:
            path = write_metrics_jsonl(records, run_dir)
            print(f"[runner] metrics.jsonl written: {path} ({len(records)} records)")
        else:
            print(f"[runner] WARN: no metrics extracted from stdout")
    except Exception as e:
        print(f"[runner] WARN: metrics extraction failed: {e}")


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
