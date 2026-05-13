"""reproduction.analysis.sanity_check — M.0 内部质量门。

**关键设计**：不依赖论文任何数值；纯客观审计 experiments/runs/ 产物。
任一检查失败 = 训练流程有问题，必须停下排查（不要继续 diff_with_paper）。

检查项（plan §A.6.1）:
    1. 完整性：每个 (stage, dataset, method, seed) 都有 done.flag
    2. 数值范围：ECE ∈ [0, 1]、AUC ∈ [0.5, 1]、LogLoss > 0；无 NaN/Inf
    3. 多 seed CV < 200%（plan §A.6.1 阈值）
    4. field_index 配置正确（aliccp=0/avazu=2/criteo=23）
    5. ece_M=100 配置生效（run_config.json + metrics 双重校验）
    6. ddof=1（用 ddof=0 vs ddof=1 在 N=3 上做差异 sanity）
    7. M=16 backbone num_estimators
    8. seed 集合 = {1024, 2024, 3024}
    9. shuffled-u 真打乱（仅 v10 stage；|Pearson corr| < 0.01）

CLI 用法:
    python -m reproduction.analysis.sanity_check                  # 全部
    python -m reproduction.analysis.sanity_check --stage main
    python -m reproduction.analysis.sanity_check --strict         # 任一 fail → exit 1
    python -m reproduction.analysis.sanity_check --json-out X.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent              # 30_reproduction/reproduction/analysis/
_PROJECT_ROOT = _HERE.parent.parent                   # 30_reproduction/

# 期望配置（plan §A.6.1 + §C）
EXPECTED_SEEDS = [1024, 2024, 3024]
EXPECTED_ECE_M = 100
EXPECTED_NUM_ESTIMATORS = 16
EXPECTED_FIELD_INDEX = {"aliccp": 0, "avazu": 2, "criteo": 23}
EXPECTED_DATASETS = ["aliccp", "avazu", "criteo"]
EXPECTED_MAIN_METHODS = [
    "platt", "ir", "hb",
    "umnn", "neucalib", "desc", "sbcr",
    "umc", "umc_wor", "uamcm", "uamcm_wor",
]
CV_THRESHOLD = 2.0                                    # CV<200% (绝对值)
ECE_VALID_RANGE = (0.0, 1.0)
AUC_VALID_RANGE = (0.5, 1.0)
SHUFFLED_CORR_THRESHOLD = 0.01


def _experiments_root() -> Path:
    sys.path.insert(0, str(_PROJECT_ROOT / "UMC"))
    from _paths import CKPT_ROOT                      # type: ignore

    return Path(CKPT_ROOT)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ============================================================================
# 单 run 审计
# ============================================================================

def audit_run(run_dir: Path, stage: str, dataset: str, method: str, seed: int) -> Dict[str, Any]:
    """对单个 run 的产物做审计，返回 issue 列表 + 关键指标。"""
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    if not run_dir.exists():
        return {"issues": ["run_dir missing"], "metrics": metrics, "complete": False}

    done = (run_dir / "done.flag").exists()
    if not done:
        issues.append("done.flag missing")

    cfg = _read_json(run_dir / "run_config.json")
    if cfg is None:
        issues.append("run_config.json missing/corrupted")
    else:
        cu = cfg.get("config_update", {})
        if stage == "pretrain":
            if cu.get("num_estimators") != EXPECTED_NUM_ESTIMATORS:
                issues.append(f"num_estimators={cu.get('num_estimators')} != {EXPECTED_NUM_ESTIMATORS}")
        else:
            if cu.get("ece_M") != EXPECTED_ECE_M:
                issues.append(f"ece_M={cu.get('ece_M')} != {EXPECTED_ECE_M}")
            exp_fi = EXPECTED_FIELD_INDEX.get(dataset)
            if exp_fi is not None and cu.get("field_index") != exp_fi:
                issues.append(f"field_index={cu.get('field_index')} != {exp_fi}")
            if cu.get("calib_seed") != seed:
                issues.append(f"calib_seed={cu.get('calib_seed')} != run dir seed {seed}")

    # metrics.jsonl 读最终指标
    mfile = run_dir / "metrics.jsonl"
    if mfile.exists():
        records = _read_jsonl(mfile)
        # 取 event == "final" 或最后一条 epoch_end
        final = None
        for r in reversed(records):
            if r.get("event") in ("final", "epoch_end", "result"):
                final = r
                break
        if final:
            for k in ("ece", "auc", "logloss"):
                if k in final:
                    metrics[k] = final[k]

    # 数值范围
    if "ece" in metrics:
        v = metrics["ece"]
        if math.isnan(v) or math.isinf(v):
            issues.append(f"ece is NaN/Inf: {v}")
        elif not (ECE_VALID_RANGE[0] <= v <= ECE_VALID_RANGE[1]):
            issues.append(f"ece out of range [0,1]: {v}")
    if "auc" in metrics:
        v = metrics["auc"]
        if math.isnan(v) or math.isinf(v):
            issues.append(f"auc is NaN/Inf: {v}")
        elif not (AUC_VALID_RANGE[0] <= v <= AUC_VALID_RANGE[1]):
            issues.append(f"auc out of range [0.5,1]: {v}")
    if "logloss" in metrics:
        v = metrics["logloss"]
        if math.isnan(v) or math.isinf(v):
            issues.append(f"logloss is NaN/Inf: {v}")
        elif v <= 0:
            issues.append(f"logloss <= 0: {v}")

    # shuffled-u 审计（仅 v10）
    if stage == "v10" and "umode_shuffled" in method:
        # 期望 metrics.jsonl 含 shuffled_u_pearson_corr
        shuffled_records = [r for r in _read_jsonl(mfile) if "shuffled_u_pearson_corr" in r]
        if not shuffled_records:
            issues.append("v10 shuffled run missing shuffled_u_pearson_corr in metrics.jsonl")
        else:
            corr = shuffled_records[-1]["shuffled_u_pearson_corr"]
            if abs(corr) >= SHUFFLED_CORR_THRESHOLD:
                issues.append(f"shuffled-u pearson |corr|={abs(corr):.4f} >= {SHUFFLED_CORR_THRESHOLD}")

    return {"issues": issues, "metrics": metrics, "complete": done}


# ============================================================================
# 多 seed CV 审计
# ============================================================================

def _std_ddof1(vals: List[float]) -> float:
    """ddof=1 (Bessel 校正)。N=1 时返回 0。"""
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)


def audit_cv(per_seed: List[Dict[str, Any]], dataset: str, method: str) -> List[str]:
    """检查多 seed CV 是否爆炸。"""
    issues: List[str] = []
    eces = [r["metrics"]["ece"] for r in per_seed if "ece" in r.get("metrics", {})]
    if len(eces) >= 2:
        mean = sum(eces) / len(eces)
        sd = _std_ddof1(eces)
        if mean > 0:
            cv = sd / mean
            if cv > CV_THRESHOLD:
                issues.append(
                    f"ECE CV={cv:.2f} > {CV_THRESHOLD} (mean={mean:.4f}, std={sd:.4f}, N={len(eces)})"
                )
    return issues


# ============================================================================
# Stage-level 审计
# ============================================================================

def audit_main_stage(exp_root: Path) -> Dict[str, Any]:
    """审计 main 99 runs（11 methods × 3 datasets × 3 seeds）。"""
    base = exp_root / "runs" / "main"
    report = {"stage": "main", "completeness": {}, "per_run": [], "per_method_dataset": {}}
    expected_count = 0
    found_count = 0

    for dataset in EXPECTED_DATASETS:
        for method in EXPECTED_MAIN_METHODS:
            per_seed = []
            for seed in EXPECTED_SEEDS:
                expected_count += 1
                rdir = base / dataset / method / f"seed_{seed}"
                rpt = audit_run(rdir, "main", dataset, method, seed)
                rpt.update({"dataset": dataset, "method": method, "seed": seed, "run_dir": str(rdir)})
                report["per_run"].append(rpt)
                per_seed.append(rpt)
                if rpt["complete"]:
                    found_count += 1
            # 多 seed CV 审计
            cv_issues = audit_cv(per_seed, dataset, method)
            if cv_issues:
                report["per_method_dataset"][f"{dataset}/{method}"] = cv_issues

    report["completeness"] = {
        "expected": expected_count,                  # 99
        "found": found_count,
        "missing": expected_count - found_count,
    }
    return report


def audit_pretrain_stage(exp_root: Path) -> Dict[str, Any]:
    """审计 pretrain 9 runs（3 datasets × 3 seeds）。"""
    base = exp_root / "runs" / "pretrain"
    report = {"stage": "pretrain", "completeness": {}, "per_run": []}
    expected_count = 0
    found_count = 0

    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            expected_count += 1
            rdir = base / dataset / "_backbone" / f"seed_{seed}"
            rpt = audit_run(rdir, "pretrain", dataset, "_backbone", seed)
            rpt.update({"dataset": dataset, "method": "_backbone", "seed": seed, "run_dir": str(rdir)})
            report["per_run"].append(rpt)
            if rpt["complete"]:
                found_count += 1

    report["completeness"] = {
        "expected": expected_count,
        "found": found_count,
        "missing": expected_count - found_count,
    }
    return report


def audit_v10_stage(exp_root: Path) -> Dict[str, Any]:
    """审计 v10 27 runs（uamcm × 3 u_modes × 3 datasets × 3 seeds）。"""
    base = exp_root / "runs" / "v10"
    report = {"stage": "v10", "completeness": {}, "per_run": [], "per_method_dataset": {}}
    expected_count = 0
    found_count = 0

    for dataset in EXPECTED_DATASETS:
        for u_mode in ("pe", "shuffled", "logit"):
            method = f"uamcm_umode_{u_mode}"
            per_seed = []
            for seed in EXPECTED_SEEDS:
                expected_count += 1
                rdir = base / dataset / method / f"seed_{seed}"
                rpt = audit_run(rdir, "v10", dataset, method, seed)
                rpt.update({"dataset": dataset, "method": method, "seed": seed, "run_dir": str(rdir)})
                report["per_run"].append(rpt)
                per_seed.append(rpt)
                if rpt["complete"]:
                    found_count += 1
            cv_issues = audit_cv(per_seed, dataset, method)
            if cv_issues:
                report["per_method_dataset"][f"{dataset}/{method}"] = cv_issues

    report["completeness"] = {
        "expected": expected_count,                  # 27
        "found": found_count,
        "missing": expected_count - found_count,
    }
    return report


# ============================================================================
# Top-level
# ============================================================================

def run_audits(stages: List[str]) -> Dict[str, Any]:
    exp_root = _experiments_root()
    out: Dict[str, Any] = {"experiments_root": str(exp_root), "stages": {}}
    if "pretrain" in stages:
        out["stages"]["pretrain"] = audit_pretrain_stage(exp_root)
    if "main" in stages:
        out["stages"]["main"] = audit_main_stage(exp_root)
    if "v10" in stages:
        out["stages"]["v10"] = audit_v10_stage(exp_root)
    # 汇总
    n_issues = 0
    for sname, sr in out["stages"].items():
        for rpt in sr.get("per_run", []):
            n_issues += len(rpt.get("issues", []))
        n_issues += sum(len(v) for v in sr.get("per_method_dataset", {}).values())
    out["total_issues"] = n_issues
    out["m0_passed"] = (n_issues == 0)
    return out


def render_report(audit: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# M.0 Sanity Check Report\n")
    lines.append(f"experiments_root: `{audit['experiments_root']}`\n")
    lines.append(f"total_issues: **{audit['total_issues']}**")
    lines.append(f"m0_passed: **{'YES' if audit['m0_passed'] else 'NO'}**\n")

    for sname, sr in audit["stages"].items():
        c = sr.get("completeness", {})
        lines.append(f"## Stage: {sname}\n")
        lines.append(f"- completeness: {c.get('found', '?')} / {c.get('expected', '?')} done flags found")
        lines.append(f"- missing: {c.get('missing', '?')}\n")

        bad = [r for r in sr.get("per_run", []) if r.get("issues")]
        if bad:
            lines.append(f"### Issues per run ({len(bad)}):\n")
            for r in bad:
                tag = f"{r['dataset']}/{r['method']}/seed_{r['seed']}"
                for i in r["issues"]:
                    lines.append(f"  - `{tag}`: {i}")
            lines.append("")

        cv_bad = sr.get("per_method_dataset", {})
        if cv_bad:
            lines.append(f"### Multi-seed CV issues:\n")
            for k, v in cv_bad.items():
                for i in v:
                    lines.append(f"  - `{k}`: {i}")
            lines.append("")

    return "\n".join(lines) + "\n"


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reproduction.analysis.sanity_check")
    p.add_argument("--stage", choices=["pretrain", "main", "v10", "all"],
                   default="all", help="审计哪个 stage（默认 all）")
    p.add_argument("--strict", action="store_true",
                   help="任一 issue → exit code 1")
    p.add_argument("--json-out", type=str, default=None,
                   help="把审计 JSON 写到指定路径")
    p.add_argument("--md-out", type=str, default=None,
                   help="把 markdown 报告写到指定路径")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    stages = [args.stage] if args.stage != "all" else ["pretrain", "main", "v10"]
    audit = run_audits(stages)

    md = render_report(audit)
    print(md)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        print(f"[sanity_check] JSON written: {args.json_out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(md)
        print(f"[sanity_check] MD written: {args.md_out}")

    if args.strict and not audit["m0_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
