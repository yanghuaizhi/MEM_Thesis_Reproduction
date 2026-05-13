"""reproduction.analysis.diff_with_paper — 四层验证 + 三态评估。

按 plan §M.2-M.6 输出 results/diff_audit/L{1,2,3,4}_*.md，最后综合
summary 报告。

复现哲学（plan §A.4）:
    本模块**不**做"数值匹配"判定。每个论断按"支持/中立/反对"三态评估，
    论文 v1.13 数值仅作"参考对比"，不作 ground truth。

判定逻辑（plan §A.4.1）:
    P1 三种误差模式可区分（L1 诊断）
    P2 UAMCM AliCCP 改善 + shuffled-u 恶化（L2 + L3 联合）
    P3 UAMCM Criteo 改善 + shuffled-u 恶化 + 统计方法压制
    P4 UAMCM Avazu shuffled-u 未显著恶化（关键反例）
    P5 诊断预判 = 实验验证（L1 + L3 一致性）
    S1/S2/S3 决策框架（L4）

CLI 用法:
    python -m reproduction.analysis.diff_with_paper --layer diagnosis
    python -m reproduction.analysis.diff_with_paper --layer method
    python -m reproduction.analysis.diff_with_paper --layer mechanism
    python -m reproduction.analysis.diff_with_paper --layer decision
    python -m reproduction.analysis.diff_with_paper --layer summary
    python -m reproduction.analysis.diff_with_paper --all          # 全部
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent


# ============================================================================
# 论文 v1.13 参考数值（plan §A.3）
# 仅作"参考对比"，不作 ground truth。
# ============================================================================

PAPER_REFERENCE = {
    "aliccp": {
        "pcoc": 1.483,
        "over_predict_bins_out_of_20": 18,
        "pcoc_cv_pct": 24.25,
        "umc_ece_mean": 8.54, "umc_ece_std": 2.09,
        "uamcm_ece_mean": 7.37, "uamcm_ece_std": 0.65,
        "uamcm_vs_umc_improvement_pct": 13.7,
        "shuffled_u_worsening_pct": 70.1,
        "pattern": "A",
    },
    "avazu": {
        "pcoc": 1.062,
        "over_predict_bins_out_of_20": 14,
        "pcoc_cv_pct": 7.73,
        "umc_ece_mean": 13.17, "umc_ece_std": 4.25,
        "uamcm_ece_mean": 10.86, "uamcm_ece_std": 4.36,
        "uamcm_vs_umc_improvement_pct": 17.6,
        "shuffled_u_worsening_pct": -7.9,           # 关键：未显著恶化
        "pattern": "C",
    },
    "criteo": {
        "pcoc": 0.962,
        "over_predict_bins_out_of_20": 0,
        "pcoc_cv_pct": 4.31,
        "umc_ece_mean": 4.83, "umc_ece_std": 1.69,
        "uamcm_ece_mean": 2.56, "uamcm_ece_std": 1.47,
        "uamcm_vs_umc_improvement_pct": 46.9,
        "shuffled_u_worsening_pct": 68.6,
        "pattern": "B",
    },
}


# ============================================================================
# 三态评估结果
# ============================================================================

@dataclass
class Verdict:
    """单个论断的判定结果。"""
    name: str
    state: str                                     # "supports" | "neutral" | "opposes" | "no_data"
    detail: str
    paper_reference: Optional[Any] = None
    reproduction_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _state_emoji(state: str) -> str:
    return {"supports": "OK", "neutral": "MID", "opposes": "OPP", "no_data": "N/A"}.get(state, "?")


# ============================================================================
# 数据加载
# ============================================================================

def _experiments_root() -> Path:
    sys.path.insert(0, str(_PROJECT_ROOT / "UMC"))
    from _paths import CKPT_ROOT                   # type: ignore

    return Path(CKPT_ROOT)


def load_main_metrics() -> List[Dict[str, Any]]:
    """读 main 99 runs 的 metrics.jsonl，返回扁平 record 列表。"""
    base = _experiments_root() / "runs" / "main"
    out: List[Dict[str, Any]] = []
    if not base.exists():
        return out
    for ds_dir in base.iterdir():
        if not ds_dir.is_dir():
            continue
        for method_dir in ds_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for seed_dir in method_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                if not (seed_dir / "done.flag").exists():
                    continue
                final = _read_final_metric(seed_dir / "metrics.jsonl")
                if not final:
                    continue
                seed = int(seed_dir.name.split("_")[1])
                out.append({
                    "dataset": ds_dir.name,
                    "method": method_dir.name,
                    "seed": seed,
                    **final,
                })
    return out


def load_v10_metrics() -> List[Dict[str, Any]]:
    """读 v10 27 runs 的 metrics.jsonl。method 列形如 'uamcm_umode_pe'。"""
    base = _experiments_root() / "runs" / "v10"
    out: List[Dict[str, Any]] = []
    if not base.exists():
        return out
    for ds_dir in base.iterdir():
        if not ds_dir.is_dir():
            continue
        for method_dir in ds_dir.iterdir():
            if not method_dir.is_dir():
                continue
            if "_umode_" not in method_dir.name:
                continue
            base_method, u_mode = method_dir.name.rsplit("_umode_", 1)
            for seed_dir in method_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                if not (seed_dir / "done.flag").exists():
                    continue
                final = _read_final_metric(seed_dir / "metrics.jsonl")
                if not final:
                    continue
                seed = int(seed_dir.name.split("_")[1])
                out.append({
                    "dataset": ds_dir.name,
                    "method": base_method,
                    "u_mode": u_mode,
                    "seed": seed,
                    **final,
                })
    return out


def load_v9_samples(dataset: str, method: str = "uamcm", seed: int = 1024) -> Optional[Dict[str, Any]]:
    """读某 (dataset, method, seed) 的 v9 sample-level NPZ。

    B3: UMC `save_samples.save_sample_level` 实际保存 4 字段:
        y_true, y_pred_uncalib, y_pred_calib, sigma2
    `u` 字段不存在，需现场从 sigma2 派生: u = log(sigma2 + 1e-8)
    （与 UMC/train_neu_ali.py L142 一致）

    路径模板（与 orchestrator._build_v9_plan 一致）:
        <CKPT_ROOT>/v9_samples/<dataset>/<method>_seed_<seed>_samples.npz
    """
    import numpy as np

    v9_dir = _experiments_root() / "v9_samples" / dataset
    target = v9_dir / f"{method}_seed_{seed}_samples.npz"
    if not target.exists():
        # fallback: 任意 NPZ
        npz_files = list(v9_dir.glob("*.npz"))
        if not npz_files:
            return None
        target = npz_files[0]
    arr = np.load(target)
    data = {k: arr[k] for k in arr.files}
    # B3: 派生 u = log(sigma2 + 1e-8)
    if "sigma2" in data and "u" not in data:
        data["u"] = np.log(data["sigma2"].astype(np.float64) + 1e-8)
    return data


def _read_final_metric(jsonl_path: Path) -> Optional[Dict[str, Any]]:
    if not jsonl_path.exists():
        return None
    last: Optional[Dict[str, Any]] = None
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") in ("final", "result", "epoch_end"):
                    last = rec
    except OSError:
        return None
    if not last:
        return None
    return {k: last[k] for k in ("ece", "auc", "logloss") if k in last}


# ============================================================================
# 聚合工具
# ============================================================================

def aggregate_mean_std(
    records: List[Dict[str, Any]],
    metric: str = "ece",
    ddof: int = 1,                                 # plan §B 第 6 条：必须 ddof=1
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """按 (dataset, method) 聚合，返回 {key: {mean, std, n}}。"""
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for r in records:
        if metric not in r:
            continue
        key = (r["dataset"], r["method"])
        grouped.setdefault(key, []).append(float(r[metric]))
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for k, vals in grouped.items():
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        if n < 2:
            std = 0.0
        else:
            var = sum((v - mean) ** 2 for v in vals) / (n - ddof)
            std = math.sqrt(var)
        out[k] = {"mean": mean, "std": std, "n": n}
    return out


def compute_pcoc(y_pred: Any, y_true: Any) -> float:
    """PCOC = E[y_pred] / E[y_true]。"""
    import numpy as np

    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    if yt.mean() == 0:
        return float("nan")
    return float(yp.mean() / yt.mean())


# ============================================================================
# F1: 从复现 v9 数据派生诊断预判（替代 hardcode 引用 PAPER_REFERENCE）
# ============================================================================

def compute_diagnosis_prediction(samples: Dict[str, Any], n_bins: int = 20) -> Dict[str, Any]:
    """从 v9 sample-level NPZ 派生 u 信号有效性预判。

    输入:
        samples: load_v9_samples() 返回（含 y_pred_uncalib, y_true, u）

    Returns dict:
        pcoc: 全局 PCOC
        over_predict_bins: per-u-bin 中 PCOC>1.0 的桶数 (/20)
        monotonic: "decreasing" | "increasing" | "neither"
        cv_pct: per-u-bin PCOC 变异系数
        pattern: "A" | "B" | "C"      # 派生模式
        u_signal_prediction: 自然语言描述
    """
    import numpy as np

    yp = np.asarray(samples["y_pred_uncalib"], dtype=float)
    yt = np.asarray(samples["y_true"], dtype=float)
    u = np.asarray(samples["u"], dtype=float)

    pcoc = compute_pcoc(yp, yt)
    per_bin = compute_per_u_bin_pcoc(yp, yt, u, n_bins=n_bins)
    valid = [p for p in per_bin["pcoc_per_bin"] if not math.isnan(p)]

    # 派生模式（按 plan §A.4.1 阈值）
    if pcoc > 1.2 and per_bin["over_predict_bins"] >= 18 and per_bin["monotonic_decreasing"]:
        pattern = "A"   # 强过预测
        prediction = "u 信号全域有效（shuffled 应显著恶化）"
    elif pcoc < 1.0 and per_bin["over_predict_bins"] <= 2 and per_bin["monotonic_decreasing"]:
        pattern = "B"   # 弱欠预测
        prediction = "u 信号局部显著（shuffled 应显著恶化）"
    elif 0.95 <= pcoc <= 1.15 and per_bin["non_monotonic"]:
        pattern = "C"   # 非单调混合
        prediction = "u 信号方向混乱（shuffled 应不显著恶化）"
    else:
        pattern = "unclassified"
        prediction = "诊断模式不属于 A/B/C 三类，u 信号预判待定"

    return {
        "pcoc": pcoc,
        "over_predict_bins": per_bin["over_predict_bins"],
        "monotonic": (
            "decreasing" if per_bin["monotonic_decreasing"]
            else "increasing" if per_bin["monotonic_increasing"]
            else "non_monotonic"
        ),
        "cv_pct": per_bin["cv_pct"],
        "pattern": pattern,
        "u_signal_prediction": prediction,
    }


# ============================================================================
# Bootstrap + paired significance (FIX-10)
# ============================================================================

def bootstrap_ece_ci(
    y_true: Any,
    y_pred: Any,
    n_bins: int = 100,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 1024,
) -> Dict[str, float]:
    """对单个 (y_true, y_pred) 做 bootstrap on test set 计算 ECE 置信区间。

    无 GPU 成本，仅 numpy。每个 bootstrap 重采样 N 个 index with replacement，重算 ECE。
    Returns: {mean, std, lo, hi}（置信区间端点）
    """
    import numpy as np

    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    n = len(yt)
    rng = np.random.RandomState(seed)

    eces: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        eces.append(_ece_fast(yt[idx], yp[idx], n_bins))
    eces_arr = np.array(eces)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(eces_arr.mean()),
        "std": float(eces_arr.std(ddof=1)),
        "lo": float(np.quantile(eces_arr, alpha)),
        "hi": float(np.quantile(eces_arr, 1 - alpha)),
    }


def paired_ece_test(
    y_true: Any,
    y_pred_a: Any,
    y_pred_b: Any,
    n_bins: int = 100,
    n_bootstrap: int = 1000,
    seed: int = 1024,
) -> Dict[str, Any]:
    """Paired bootstrap test: 两个模型的 ECE 差异是否显著。

    每个 bootstrap 用同一组 sample indices 重算两个模型的 ECE，得到差异分布。
    Returns: {ece_a_mean, ece_b_mean, diff_mean, diff_lo, diff_hi, p_value (one-sided), significant}
    """
    import numpy as np

    yt = np.asarray(y_true, dtype=float).reshape(-1)
    ya = np.asarray(y_pred_a, dtype=float).reshape(-1)
    yb = np.asarray(y_pred_b, dtype=float).reshape(-1)
    n = len(yt)
    rng = np.random.RandomState(seed)

    diffs: List[float] = []
    ecesa: List[float] = []
    ecesb: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        e_a = _ece_fast(yt[idx], ya[idx], n_bins)
        e_b = _ece_fast(yt[idx], yb[idx], n_bins)
        ecesa.append(e_a)
        ecesb.append(e_b)
        diffs.append(e_a - e_b)
    diff_arr = np.array(diffs)
    # one-sided p-value: P(diff ≤ 0)（"b 比 a 好"的零假设反对率）
    p_value = float((diff_arr <= 0).mean())
    return {
        "ece_a_mean": float(np.mean(ecesa)),
        "ece_b_mean": float(np.mean(ecesb)),
        "diff_mean": float(diff_arr.mean()),
        "diff_lo": float(np.quantile(diff_arr, 0.025)),
        "diff_hi": float(np.quantile(diff_arr, 0.975)),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def _ece_fast(y_true: Any, y_pred: Any, n_bins: int = 100) -> float:
    """快速 ECE 计算（与 UMC/utils/metric.py get_ece 数学一致，但纯 numpy）。"""
    import numpy as np

    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if len(yt) == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(yp, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        acc = yt[mask].mean()
        conf = yp[mask].mean()
        ece += (mask.sum() / len(yt)) * abs(acc - conf)
    return float(ece)



def compute_per_u_bin_pcoc(
    y_pred: Any,
    y_true: Any,
    u: Any,
    n_bins: int = 20,
) -> Dict[str, Any]:
    """按 u 分位分桶计算 PCOC 序列。"""
    import numpy as np

    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    u = np.asarray(u, dtype=float)

    if len(u) == 0:
        return {"bins": [], "pcoc_per_bin": [], "over_predict_bins": 0}

    edges = np.quantile(u, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9                              # 包含右端
    bin_ids = np.digitize(u, edges[1:-1])

    pcoc_per_bin: List[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            pcoc_per_bin.append(float("nan"))
            continue
        ypb = yp[mask].mean()
        ytb = yt[mask].mean()
        pcoc_per_bin.append(float("nan") if ytb == 0 else float(ypb / ytb))

    valid = [p for p in pcoc_per_bin if not math.isnan(p)]
    over_predict = sum(1 for p in valid if p > 1.0)
    monotonic = _check_monotonic(valid)
    return {
        "bins": list(range(n_bins)),
        "pcoc_per_bin": pcoc_per_bin,
        "over_predict_bins": over_predict,
        "monotonic_decreasing": monotonic == "decreasing",
        "monotonic_increasing": monotonic == "increasing",
        "non_monotonic": monotonic == "neither",
        "cv_pct": _cv_pct(valid),
    }


def _check_monotonic(seq: List[float]) -> str:
    if len(seq) < 2:
        return "neither"
    diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    if all(d <= 1e-6 for d in diffs):
        return "decreasing"
    if all(d >= -1e-6 for d in diffs):
        return "increasing"
    return "neither"


def _cv_pct(vals: List[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean == 0:
        return 0.0
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
    return 100.0 * std / abs(mean)


# ============================================================================
# L1: 诊断层（P1）
# ============================================================================

def check_p1_diagnosis() -> Dict[str, Verdict]:
    """P1: 三种误差模式可区分。"""
    results: Dict[str, Verdict] = {}
    for dataset, ref in PAPER_REFERENCE.items():
        samples = load_v9_samples(dataset)
        if samples is None:
            results[dataset] = Verdict(
                name=f"P1 [{dataset}]",
                state="no_data",
                detail="v9 samples NPZ not yet generated",
                paper_reference={"pcoc": ref["pcoc"], "pattern": ref["pattern"]},
            )
            continue

        y_pred = samples.get("y_pred_uncalib")
        y_true = samples.get("y_true")
        u = samples.get("u")
        if y_pred is None or y_true is None or u is None:
            results[dataset] = Verdict(
                name=f"P1 [{dataset}]",
                state="no_data",
                detail="NPZ missing required fields",
            )
            continue

        pcoc = compute_pcoc(y_pred, y_true)
        per_bin = compute_per_u_bin_pcoc(y_pred, y_true, u, n_bins=20)
        details = []
        ok_count = 0
        total = 0
        if dataset == "aliccp":
            total = 4
            ok_count += int(pcoc > 1.2)
            details.append(f"pcoc={pcoc:.3f} > 1.2: {pcoc > 1.2}")
            ok_count += int(per_bin["over_predict_bins"] >= 18)
            details.append(f"over_predict_bins={per_bin['over_predict_bins']} >= 18: {per_bin['over_predict_bins'] >= 18}")
            ok_count += int(per_bin["monotonic_decreasing"])
            details.append(f"monotonic_decreasing: {per_bin['monotonic_decreasing']}")
            ok_count += int(per_bin["cv_pct"] > 15)
            details.append(f"cv_pct={per_bin['cv_pct']:.2f}% > 15: {per_bin['cv_pct'] > 15}")
        elif dataset == "criteo":
            total = 3
            ok_count += int(0.90 <= pcoc < 1.0)
            details.append(f"pcoc={pcoc:.3f} in [0.90, 1.0): {0.90 <= pcoc < 1.0}")
            ok_count += int(per_bin["over_predict_bins"] <= 2)
            details.append(f"over_predict_bins={per_bin['over_predict_bins']} <= 2: {per_bin['over_predict_bins'] <= 2}")
            ok_count += int(per_bin["monotonic_decreasing"])
            details.append(f"monotonic_decreasing: {per_bin['monotonic_decreasing']}")
        elif dataset == "avazu":
            total = 2
            ok_count += int(0.95 <= pcoc <= 1.15)
            details.append(f"pcoc={pcoc:.3f} in [0.95, 1.15]: {0.95 <= pcoc <= 1.15}")
            ok_count += int(per_bin["non_monotonic"])
            details.append(f"non_monotonic (方向反转): {per_bin['non_monotonic']}")

        if ok_count == total:
            state = "supports"
        elif ok_count >= total - 1:
            state = "neutral"
        else:
            state = "opposes"

        results[dataset] = Verdict(
            name=f"P1 [{dataset}]",
            state=state,
            detail=f"{ok_count}/{total} sub-checks pass; " + "; ".join(details),
            paper_reference={"pcoc": ref["pcoc"], "pattern": ref["pattern"]},
            reproduction_value={"pcoc": pcoc, "over_predict_bins": per_bin["over_predict_bins"]},
        )
    return results


# ============================================================================
# L2: 方法层（P2/P3/P4）
# ============================================================================

def check_p2_p3_p4(main_records: List[Dict[str, Any]]) -> Dict[str, Verdict]:
    """P2 AliCCP, P3 Criteo, P4 Avazu — UAMCM 改善幅度判定。"""
    agg = aggregate_mean_std(main_records, "ece", ddof=1)
    results: Dict[str, Verdict] = {}

    for tag, dataset, support_imp, oppose_imp, neutral_low in [
        ("P2", "aliccp", 5.0, 0.0, 1.0),       # 改善 ≥ 5% 支持；反方向反对；1-5% 中立
        ("P3", "criteo", 25.0, 0.0, 10.0),     # 改善 ≥ 25% 支持；10-25% 中立
        ("P4", "avazu", None, None, None),     # 仅检查 ECE 改善方向（详细 P4 见 L3）
    ]:
        umc = agg.get((dataset, "umc"))
        uamcm = agg.get((dataset, "uamcm"))
        if not umc or not uamcm:
            results[tag] = Verdict(
                name=f"{tag} [{dataset}]",
                state="no_data",
                detail=f"main metrics missing (umc={bool(umc)}, uamcm={bool(uamcm)})",
            )
            continue
        improvement_pct = 100 * (umc["mean"] - uamcm["mean"]) / umc["mean"] if umc["mean"] else 0
        if tag == "P4":
            # P4 仅判 ECE 改善方向（详细 shuffled-u 在 L3）
            state = "supports" if improvement_pct > 0 else "opposes"
            detail = f"UAMCM improvement {improvement_pct:.1f}% (UMC ECE {umc['mean']*100:.2f}, UAMCM {uamcm['mean']*100:.2f})"
        else:
            if improvement_pct >= support_imp:
                state = "supports"
            elif improvement_pct <= oppose_imp:
                state = "opposes"
            elif neutral_low and improvement_pct >= neutral_low:
                state = "neutral"
            else:
                state = "opposes"
            detail = (
                f"UAMCM improvement {improvement_pct:.1f}% "
                f"(UMC ECE {umc['mean']*100:.2f}±{umc['std']*100:.2f}, "
                f"UAMCM ECE {uamcm['mean']*100:.2f}±{uamcm['std']*100:.2f}); "
                f"thresholds: support≥{support_imp}%, oppose≤{oppose_imp}%"
            )
        results[tag] = Verdict(
            name=f"{tag} [{dataset}]",
            state=state,
            detail=detail,
            paper_reference={
                "improvement_pct": PAPER_REFERENCE[dataset]["uamcm_vs_umc_improvement_pct"],
            },
            reproduction_value={"improvement_pct": improvement_pct,
                               "umc_ece": umc["mean"],
                               "uamcm_ece": uamcm["mean"]},
        )
    return results


# ============================================================================
# L3: 机制层（P5 shuffled-u）
# ============================================================================

def check_p5_diagnosis_vs_validation() -> Dict[str, Verdict]:
    """F1 派生预判 vs v10 实验验证 一致性（P5 论证支点）。

    流程:
        1. 对每个 dataset，从 v9 NPZ 计算诊断预判（pattern + u 有效性）
        2. 对每个 dataset，从 v10 实验结果判定 shuffled 是否恶化
        3. 一致性: 派生预判方向 == v10 验证方向？

    **独立性标注**: v9 派生 + v10 验证基于同一 model（UAMCM × seed=1024），
    非完全独立。该函数仅验证"复现内部 v9/v10 一致性"，不能替代用不同 model 做的
    真独立验证。
    """
    v10_records = load_v10_metrics()
    by_key = aggregate_mean_std(
        [{**r, "method": r["u_mode"]} for r in v10_records],
        "ece", ddof=1,
    )

    results: Dict[str, Verdict] = {}
    for dataset in ("aliccp", "avazu", "criteo"):
        # 1. v9 派生预判
        samples = load_v9_samples(dataset, method="uamcm", seed=1024)
        if samples is None or "u" not in samples:
            results[f"P5_diagnosis_{dataset}"] = Verdict(
                name=f"P5 派生预判 vs 实验 [{dataset}]",
                state="no_data",
                detail="v9 samples NPZ 缺失或 u 字段未派生",
            )
            continue
        derived = compute_diagnosis_prediction(samples)
        predicted_shuffled_worsens = derived["pattern"] in ("A", "B")  # A/B 预判恶化，C 预判不恶化

        # 2. v10 实验
        pe = by_key.get((dataset, "pe"))
        shuf = by_key.get((dataset, "shuffled"))
        if not pe or not shuf:
            results[f"P5_diagnosis_{dataset}"] = Verdict(
                name=f"P5 派生预判 vs 实验 [{dataset}]",
                state="no_data",
                detail=f"v10 数据缺失（pe={bool(pe)}, shuf={bool(shuf)})",
                reproduction_value={"derived": derived},
            )
            continue
        worsening_pct = 100 * (shuf["mean"] - pe["mean"]) / pe["mean"] if pe["mean"] else 0
        if dataset == "avazu":
            sigma_pct = 100 * shuf["std"] / pe["mean"] if pe["mean"] else 0
            actual_shuffled_worsens = abs(worsening_pct) > sigma_pct
        else:
            actual_shuffled_worsens = worsening_pct >= 30

        # 3. 一致性
        if predicted_shuffled_worsens == actual_shuffled_worsens:
            state = "supports"
            detail = (
                f"派生预判 (pattern={derived['pattern']}: {derived['u_signal_prediction']}) "
                f"与 v10 实验方向一致 (shuffled 恶化 {worsening_pct:+.1f}%)"
            )
        else:
            state = "opposes"
            detail = (
                f"派生预判 (pattern={derived['pattern']}: {derived['u_signal_prediction']}) "
                f"与 v10 实验方向**矛盾** (shuffled 恶化 {worsening_pct:+.1f}%)"
            )
        results[f"P5_diagnosis_{dataset}"] = Verdict(
            name=f"P5 派生预判 vs 实验 [{dataset}]",
            state=state,
            detail=detail,
            reproduction_value={
                "derived_pattern": derived["pattern"],
                "derived_pcoc": derived["pcoc"],
                "shuffled_worsening_pct": worsening_pct,
            },
        )
    return results


def check_p5_shuffled(v10_records: List[Dict[str, Any]]) -> Dict[str, Verdict]:
    """P5: 三数据集 shuffled-u 结果与诊断预判一致性 + logit 对照 (FIX-11)。

    NOTE: v9 派生预判 + v10 验证基于同一 model（UAMCM × PackedDeepFM × seed=1024），
    **非完全独立验证**——这是设计层局限，在 P5 报告中需明示。
    """
    by_key = aggregate_mean_std(
        [{**r, "method": r["u_mode"]} for r in v10_records],
        "ece", ddof=1,
    )
    results: Dict[str, Verdict] = {}

    for dataset in ("aliccp", "criteo", "avazu"):
        # R1: pe 从 main_99 取（v10 不再跑 pe）— 但 diff_with_paper 输入是 v10，
        # 这里如果 pe 数据缺，需 fallback 从 main_99 拉。最简实现：让 v10_records
        # 在上游已合并 main pe → 这里直接读 pe key
        pe = by_key.get((dataset, "pe"))
        shuf = by_key.get((dataset, "shuffled"))
        logit = by_key.get((dataset, "logit"))    # FIX-11: logit 对照
        if not pe or not shuf:
            results[f"P5_{dataset}"] = Verdict(
                name=f"P5 [{dataset}] shuffled-u",
                state="no_data",
                detail=f"v10 missing (pe={bool(pe)}, shuffled={bool(shuf)})",
            )
            continue
        worsening_pct = 100 * (shuf["mean"] - pe["mean"]) / pe["mean"] if pe["mean"] else 0
        sigma = shuf["std"]
        sigma_pct = 100 * sigma / pe["mean"] if pe["mean"] else 0

        # FIX-11: logit 对照——u 信号源敏感性测试
        logit_detail = ""
        if logit:
            logit_change_pct = 100 * (logit["mean"] - pe["mean"]) / pe["mean"] if pe["mean"] else 0
            if abs(logit_change_pct) < 10:
                logit_verdict = "u 信号源不敏感（sigma² 可被 logit 替代）"
            else:
                logit_verdict = "sigma² 是关键信号（logit 替代后显著变化）"
            logit_detail = (
                f"; logit-pe 变化 {logit_change_pct:+.1f}% → {logit_verdict}"
            )

        if dataset == "avazu":
            # P4 关键：变化在 ±σ 内 = 支持论断
            in_sigma = abs(worsening_pct) <= sigma_pct
            if in_sigma:
                state = "supports"
            elif abs(worsening_pct) <= 2 * sigma_pct:
                state = "neutral"
            else:
                state = "opposes"
            detail = (
                f"shuffled-u change {worsening_pct:+.1f}% "
                f"(σ={sigma_pct:.1f}%; in_sigma={in_sigma})"
                f"{logit_detail}"
            )
        else:
            # P2/P3 关键：shuffled-u 恶化 ≥ 30%
            if worsening_pct >= 30:
                state = "supports"
            elif worsening_pct >= 15:
                state = "neutral"
            else:
                state = "opposes"
            detail = (
                f"shuffled-u worsening {worsening_pct:+.1f}% "
                f"(PE ECE {pe['mean']*100:.2f}, shuffled {shuf['mean']*100:.2f}); "
                f"threshold ≥30%{logit_detail}"
            )
        results[f"P5_{dataset}"] = Verdict(
            name=f"P5 [{dataset}] shuffled-u",
            state=state,
            detail=detail,
            paper_reference={"shuffled_u_worsening_pct": PAPER_REFERENCE[dataset]["shuffled_u_worsening_pct"]},
            reproduction_value={
                "worsening_pct": worsening_pct,
                "logit_change_pct": (100 * (logit["mean"] - pe["mean"]) / pe["mean"]
                                     if logit and pe["mean"] else None),
            },
        )
    return results


# ============================================================================
# L4: 决策层（S1/S2/S3 + 三重门槛）
# ============================================================================

def check_decision_framework(
    main_records: List[Dict[str, Any]],
    v10_records: List[Dict[str, Any]],
) -> Dict[str, Verdict]:
    """S1=Criteo, S2=AliCCP, S3=Avazu。"""
    agg_main = aggregate_mean_std(main_records, "ece", ddof=1)
    results: Dict[str, Verdict] = {}

    # S1: Criteo 上统计方法首选（IR/Platt/HB 排前 3）
    criteo_methods = {m: v for (d, m), v in agg_main.items() if d == "criteo"}
    if criteo_methods:
        sorted_methods = sorted(criteo_methods.items(), key=lambda kv: kv[1]["mean"])
        top3 = [m for m, _ in sorted_methods[:3]]
        sta = sum(1 for m in top3 if m in ("ir", "platt", "hb"))
        if sta >= 1:
            state = "supports"
        else:
            state = "opposes"
        results["S1"] = Verdict(
            name="S1 Criteo 统计方法首选",
            state=state,
            detail=f"top3 by ECE: {top3}; statistical hits={sta}/3",
            reproduction_value={"top3": top3},
        )
    else:
        results["S1"] = Verdict(name="S1", state="no_data", detail="criteo main metrics empty")

    # S2: AliCCP 上 UAMCM 谨慎推荐（改善 + seed 一致性 ≥ 1/3）
    aliccp_umc = agg_main.get(("aliccp", "umc"))
    aliccp_uamcm = agg_main.get(("aliccp", "uamcm"))
    if aliccp_umc and aliccp_uamcm:
        imp_pct = 100 * (aliccp_umc["mean"] - aliccp_uamcm["mean"]) / aliccp_umc["mean"]
        # 单 seed 一致性需读 raw records
        seeds_winning = 0
        for s in (1024, 2024, 3024):
            umc_e = next((r["ece"] for r in main_records
                         if r["dataset"] == "aliccp" and r["method"] == "umc"
                         and r["seed"] == s and "ece" in r), None)
            uamcm_e = next((r["ece"] for r in main_records
                           if r["dataset"] == "aliccp" and r["method"] == "uamcm"
                           and r["seed"] == s and "ece" in r), None)
            if umc_e is not None and uamcm_e is not None and uamcm_e < umc_e:
                seeds_winning += 1
        if imp_pct > 0 and seeds_winning >= 1:
            state = "supports"
        elif imp_pct > 0:
            state = "neutral"
        else:
            state = "opposes"
        results["S2"] = Verdict(
            name="S2 AliCCP UAMCM 谨慎推荐",
            state=state,
            detail=f"improvement {imp_pct:.1f}%, seed_consistency {seeds_winning}/3",
            reproduction_value={"improvement_pct": imp_pct, "seeds_winning": seeds_winning},
        )
    else:
        results["S2"] = Verdict(name="S2", state="no_data", detail="aliccp umc/uamcm metrics missing")

    # S3: Avazu 上不引入 u（shuffled-u 不显著恶化）
    p5_avazu = check_p5_shuffled(v10_records).get("P5_avazu")
    if p5_avazu and p5_avazu.state != "no_data":
        results["S3"] = Verdict(
            name="S3 Avazu 不引入 u",
            state=p5_avazu.state,                  # 与 P4 同状态
            detail=f"derived from P5 [avazu]: {p5_avazu.detail}",
            reproduction_value=p5_avazu.reproduction_value,
        )
    else:
        results["S3"] = Verdict(name="S3", state="no_data", detail="v10 avazu missing")

    return results


# ============================================================================
# Report rendering
# ============================================================================

def render_layer(layer: str, verdicts: Dict[str, Verdict]) -> str:
    lines: List[str] = [f"# Layer {layer} verification\n"]
    summary = {"supports": 0, "neutral": 0, "opposes": 0, "no_data": 0}
    for v in verdicts.values():
        summary[v.state] = summary.get(v.state, 0) + 1
    lines.append(
        f"summary: supports={summary['supports']}, neutral={summary['neutral']}, "
        f"opposes={summary['opposes']}, no_data={summary['no_data']}\n"
    )
    for v in verdicts.values():
        lines.append(f"## {_state_emoji(v.state)} {v.name}: **{v.state}**")
        lines.append(f"  - detail: {v.detail}")
        if v.paper_reference:
            lines.append(f"  - paper_v1.13: {v.paper_reference}")
        if v.reproduction_value:
            lines.append(f"  - reproduction: {v.reproduction_value}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_summary(all_verdicts: Dict[str, Dict[str, Verdict]]) -> str:
    lines: List[str] = ["# Diff with paper v1.13 — Final Summary\n"]
    # 总计
    totals = {"supports": 0, "neutral": 0, "opposes": 0, "no_data": 0}
    for layer, vs in all_verdicts.items():
        for v in vs.values():
            totals[v.state] += 1
    n = sum(totals.values())
    lines.append(f"Total: {n} verdicts")
    lines.append(f"  - supports: {totals['supports']}")
    lines.append(f"  - neutral:  {totals['neutral']}")
    lines.append(f"  - opposes:  {totals['opposes']}")
    lines.append(f"  - no_data:  {totals['no_data']}\n")

    # 决策树
    overall = "全通过"
    if totals["opposes"] > totals["supports"]:
        overall = "多数反对 → 优先怀疑复现配置错误（plan §A.4.1 决策）"
    elif totals["opposes"] > 0:
        overall = f"少数反对 ({totals['opposes']}) → 根因分析"
    elif totals["neutral"] > totals["supports"]:
        overall = "多数中立 → 论文部分章节需弱化表述"
    lines.append(f"**Overall**: {overall}\n")

    for layer, vs in all_verdicts.items():
        lines.append(f"\n## {layer}")
        for v in vs.values():
            lines.append(f"  - {_state_emoji(v.state)} `{v.name}`: {v.state}")
    return "\n".join(lines) + "\n"


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reproduction.analysis.diff_with_paper")
    p.add_argument(
        "--layer",
        choices=["diagnosis", "method", "mechanism", "decision", "summary"],
        help="要跑哪一层（不指定 = --all）",
    )
    p.add_argument("--all", action="store_true", help="跑全部 4 层 + summary")
    p.add_argument("--out-dir", type=str,
                   default=str(_PROJECT_ROOT / "results" / "diff_audit"),
                   help="输出目录")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.layer and not args.all:
        args.all = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_recs = load_main_metrics()
    v10_recs = load_v10_metrics()
    all_v: Dict[str, Dict[str, Verdict]] = {}

    if args.all or args.layer == "diagnosis":
        v = check_p1_diagnosis()
        all_v["L1_diagnosis"] = v
        (out_dir / "L1_diagnosis_verification.md").write_text(render_layer("L1 (P1)", v))
        print(f"L1 written: {out_dir / 'L1_diagnosis_verification.md'}")
    if args.all or args.layer == "method":
        v = check_p2_p3_p4(main_recs)
        all_v["L2_method"] = v
        (out_dir / "L2_method_verification.md").write_text(render_layer("L2 (P2/P3/P4)", v))
        print(f"L2 written: {out_dir / 'L2_method_verification.md'}")
    if args.all or args.layer == "mechanism":
        v = check_p5_shuffled(v10_recs)
        # FIX-12: F1 派生预判 vs 实验验证 一致性（合并到 L3 mechanism）
        v.update(check_p5_diagnosis_vs_validation())
        all_v["L3_mechanism"] = v
        l3_text = render_layer("L3 (P5 + 派生预判)", v)
        l3_text += (
            "\n---\n\n## 独立性标注（plan §A.4 复现哲学）\n\n"
            "v9 派生预判 + v10 实验验证均基于**同一 model**（UAMCM × PackedDeepFM × seed=1024），"
            "因此本节命中率验证的是\"复现内部 v9/v10 一致性\"，**不是用不同 model 的真独立验证**。"
            "如需真独立，应用 model A 派生预判 + model B 验证（超出当前 reproduction 范围）。\n"
        )
        (out_dir / "L3_mechanism_verification.md").write_text(l3_text)
        print(f"L3 written: {out_dir / 'L3_mechanism_verification.md'}")
    if args.all or args.layer == "decision":
        v = check_decision_framework(main_recs, v10_recs)
        all_v["L4_decision"] = v
        (out_dir / "L4_decision_verification.md").write_text(render_layer("L4 (S1/S2/S3)", v))
        print(f"L4 written: {out_dir / 'L4_decision_verification.md'}")
    if args.all or args.layer == "summary":
        # summary 需要全部 4 层产物
        if len(all_v) < 4:
            all_v.setdefault("L1_diagnosis", check_p1_diagnosis())
            all_v.setdefault("L2_method", check_p2_p3_p4(main_recs))
            l3 = check_p5_shuffled(v10_recs)
            l3.update(check_p5_diagnosis_vs_validation())
            all_v.setdefault("L3_mechanism", l3)
            all_v.setdefault("L4_decision", check_decision_framework(main_recs, v10_recs))
        (out_dir / "diff_with_v1_13.md").write_text(render_summary(all_v))
        print(f"Summary written: {out_dir / 'diff_with_v1_13.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
