#!/usr/bin/env python
"""V6 Phase 1: Comprehensive data analysis.

Extracts metrics from all 28 experiment logs, builds unified summary,
and generates ranking tables + key analysis for paper direction decisions.
"""

import os
import re
import csv
import json

LOG_DIR = "/root/shared-nvme/PAPER/ckpt/v6_phase1/logs"
OUT_DIR = "/root/shared-nvme/PAPER/ckpt/v6_phase1"

METHODS_ORDER = [
    "sta_ir", "sta_hb", "sta_platt",
    "neu", "desc", "sbcr", "umnn",
    "umc_wor", "umc",
    "uamcm_wor", "uamcm",
    "uasac_K3", "uasac_r_K3",
    "uamcm_phase4",
]

DATASETS = ["aliccp", "avazu"]

METRIC_FIELDS = [
    "auc", "gauc", "logloss", "pcoc", "ece",
    "fece", "mfece", "rce", "frce", "mfrce",
]


def extract_metrics_from_log(log_path, tag="calibrated"):
    """Extract metrics following an exact metrics_tag=<tag> line."""
    metrics = {}
    found_tag = False
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == f"metrics_tag={tag}":
                found_tag = True
                continue
            if found_tag and "test_" in line:
                for m in re.finditer(r"test_(\w+)\s*=\s*([\d.eE+-]+)", line):
                    metrics[m.group(1)] = float(m.group(2))
                break
    return metrics


def extract_scl_params(log_path):
    """Extract scl_lam and scl_beta from log."""
    with open(log_path, "r") as f:
        for line in f:
            if "scl_params" in line:
                m_lam = re.search(r"scl_lam=([\d.eE+-]+)", line)
                m_beta = re.search(r"scl_beta=([\d.eE+-]+)", line)
                return (
                    float(m_lam.group(1)) if m_lam else None,
                    float(m_beta.group(1)) if m_beta else None,
                )
    return None, None


def extract_best_ece_epoch(log_path):
    """Extract the epoch that achieved best ECE."""
    best_ece = float("inf")
    best_epoch = None
    with open(log_path, "r") as f:
        for line in f:
            m = re.match(r"ece_track epoch=(\d+) test_ece=([\d.]+) best_ece=([\d.]+) best_ece_epoch=(\d+)", line.strip())
            if m:
                best_epoch = int(m.group(4))
                best_ece = float(m.group(3))
    return best_epoch, best_ece


def extract_early_stop(log_path):
    """Check if early stopping triggered and at which epoch."""
    last_epoch = 0
    with open(log_path, "r") as f:
        for line in f:
            m = re.match(r"calib_epoch_end epoch=(\d+)/(\d+)", line.strip())
            if m:
                last_epoch = int(m.group(1))
    return last_epoch


def build_unified_summary():
    """Build unified summary from all 28 logs."""
    rows = []

    for dataset in DATASETS:
        for method_name in METHODS_ORDER:
            log_file = os.path.join(LOG_DIR, f"{dataset}_{method_name}.log")
            if not os.path.exists(log_file):
                print(f"WARNING: Missing log: {log_file}")
                continue

            row = {"dataset": dataset, "name": method_name}

            # SCL params
            scl_lam, scl_beta = extract_scl_params(log_file)
            row["scl_lam"] = scl_lam
            row["scl_beta"] = scl_beta

            # Last-epoch (loss-best) metrics
            cal_metrics = extract_metrics_from_log(log_file, "calibrated")
            for k, v in cal_metrics.items():
                row[k] = v

            # ECE-best metrics (neural only)
            ece_metrics = extract_metrics_from_log(log_file, "ece_best_calibrated")
            for k, v in ece_metrics.items():
                row[f"ece_best_{k}"] = v

            # Best ECE epoch
            best_epoch, best_ece = extract_best_ece_epoch(log_file)
            row["best_ece_epoch"] = best_epoch

            # Last training epoch
            last_epoch = extract_early_stop(log_file)
            row["last_train_epoch"] = last_epoch

            rows.append(row)

    return rows


def print_ranking_table(rows, dataset, metric, ascending=True, title=None):
    """Print a ranking table for a given dataset and metric."""
    ds_rows = [r for r in rows if r["dataset"] == dataset and metric in r]
    ds_rows.sort(key=lambda r: r[metric], reverse=not ascending)

    title = title or f"{dataset.upper()} — Ranked by {metric} ({'↑ lower=better' if ascending else '↓ higher=better'})"
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Rank':<5} {'Method':<18} {metric:<12} {'PCOC':<10} {'|PCOC-1|':<10} {'LogLoss':<12} {'AUC':<10}")
    print(f"  {'-'*5} {'-'*18} {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for i, r in enumerate(ds_rows, 1):
        pcoc = r.get("pcoc", 0)
        pcoc_dev = abs(pcoc - 1.0)
        print(f"  {i:<5} {r['name']:<18} {r[metric]:<12.6f} {pcoc:<10.6f} {pcoc_dev:<10.6f} {r.get('logloss', 0):<12.6f} {r.get('auc', 0):<10.6f}")


def print_ece_best_ranking(rows, dataset):
    """Print ECE-best ranking (neural methods only)."""
    ds_rows = [r for r in rows if r["dataset"] == dataset and "ece_best_ece" in r]
    ds_rows.sort(key=lambda r: r["ece_best_ece"])

    print(f"\n{'='*90}")
    print(f"  {dataset.upper()} — ECE-Best Epoch Ranking (neural methods only)")
    print(f"{'='*90}")
    print(f"  {'Rank':<5} {'Method':<18} {'ECE_best':<12} {'ECE_best_ep':<12} {'PCOC_best':<12} {'LL_best':<12} {'AUC_best':<12}")
    print(f"  {'-'*5} {'-'*18} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for i, r in enumerate(ds_rows, 1):
        print(f"  {i:<5} {r['name']:<18} {r['ece_best_ece']:<12.6f} {str(r.get('best_ece_epoch', 'N/A')):<12} {r.get('ece_best_pcoc', 0):<12.6f} {r.get('ece_best_logloss', 0):<12.6f} {r.get('ece_best_auc', 0):<12.6f}")


def pcoc_ece_correlation(rows, dataset):
    """Compute correlation between |PCOC-1| and ECE."""
    ds_rows = [r for r in rows if r["dataset"] == dataset and "ece" in r and "pcoc" in r]
    pcoc_devs = [abs(r["pcoc"] - 1.0) for r in ds_rows]
    eces = [r["ece"] for r in ds_rows]

    n = len(pcoc_devs)
    if n < 3:
        return None

    mean_x = sum(pcoc_devs) / n
    mean_y = sum(eces) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(pcoc_devs, eces))
    var_x = sum((x - mean_x) ** 2 for x in pcoc_devs)
    var_y = sum((y - mean_y) ** 2 for y in eces)

    if var_x == 0 or var_y == 0:
        return None
    r = cov / (var_x * var_y) ** 0.5
    return r


def cross_dataset_consistency(rows):
    """Compare rankings across datasets."""
    print(f"\n{'='*90}")
    print(f"  CROSS-DATASET CONSISTENCY ANALYSIS")
    print(f"{'='*90}")

    for metric, asc in [("ece", True), ("logloss", True), ("auc", False)]:
        print(f"\n  --- {metric} ranking comparison ---")
        print(f"  {'Method':<18} {'AliCCP_rank':<14} {'AliCCP_val':<14} {'Avazu_rank':<14} {'Avazu_val':<14} {'Rank_diff':<10}")
        print(f"  {'-'*18} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")

        ali_rows = sorted(
            [r for r in rows if r["dataset"] == "aliccp" and metric in r],
            key=lambda r: r[metric], reverse=not asc,
        )
        ava_rows = sorted(
            [r for r in rows if r["dataset"] == "avazu" and metric in r],
            key=lambda r: r[metric], reverse=not asc,
        )

        ali_rank = {r["name"]: (i + 1, r[metric]) for i, r in enumerate(ali_rows)}
        ava_rank = {r["name"]: (i + 1, r[metric]) for i, r in enumerate(ava_rows)}

        all_methods = [m for m in METHODS_ORDER if m in ali_rank and m in ava_rank]
        for m in all_methods:
            ar, av_val = ali_rank[m]
            vr, vv_val = ava_rank[m]
            print(f"  {m:<18} {ar:<14} {av_val:<14.6f} {vr:<14} {vv_val:<14.6f} {abs(ar - vr):<10}")


def key_questions_analysis(rows):
    """Answer the 4 key Phase 1 questions."""
    print(f"\n{'='*90}")
    print(f"  KEY PHASE 1 QUESTIONS — ANSWERS")
    print(f"{'='*90}")

    for dataset in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == dataset]

        # Q1: Is UAMCM still #1 with lam=1e-2?
        ece_sorted = sorted([r for r in ds_rows if "ece" in r], key=lambda r: r["ece"])
        ece_best_sorted = sorted([r for r in ds_rows if "ece_best_ece" in r], key=lambda r: r["ece_best_ece"])
        ll_sorted = sorted([r for r in ds_rows if "logloss" in r], key=lambda r: r["logloss"])

        print(f"\n  [{dataset.upper()}]")
        print(f"  Q1: ECE ranking top-5 (loss-best):")
        for i, r in enumerate(ece_sorted[:5], 1):
            print(f"      #{i} {r['name']:<18} ECE={r['ece']:.6f}  PCOC={r.get('pcoc',0):.6f}")

        if ece_best_sorted:
            print(f"  Q1: ECE ranking top-5 (ece-best):")
            for i, r in enumerate(ece_best_sorted[:5], 1):
                print(f"      #{i} {r['name']:<18} ECE={r['ece_best_ece']:.6f}  PCOC={r.get('ece_best_pcoc',0):.6f}")

        print(f"  Q1: LogLoss ranking top-5:")
        for i, r in enumerate(ll_sorted[:5], 1):
            print(f"      #{i} {r['name']:<18} LL={r['logloss']:.6f}")

        # Q2: UASAC_R vs UAMCM gap
        uasac_r = next((r for r in ds_rows if r["name"] == "uasac_r_K3"), None)
        uamcm = next((r for r in ds_rows if r["name"] == "uamcm"), None)
        uasac = next((r for r in ds_rows if r["name"] == "uasac_K3"), None)
        if uasac_r and uamcm:
            print(f"\n  Q2: UASAC_R vs UAMCM:")
            print(f"      uasac_r_K3  ECE={uasac_r['ece']:.6f}  LL={uasac_r['logloss']:.6f}  PCOC={uasac_r['pcoc']:.6f}")
            print(f"      uamcm       ECE={uamcm['ece']:.6f}  LL={uamcm['logloss']:.6f}  PCOC={uamcm['pcoc']:.6f}")
            if uasac:
                print(f"      uasac_K3    ECE={uasac['ece']:.6f}  LL={uasac['logloss']:.6f}  PCOC={uasac['pcoc']:.6f}")
                ece_gap_before = uasac["ece"] - uamcm["ece"]
                ece_gap_after = uasac_r["ece"] - uamcm["ece"]
                print(f"      UASAC→UAMCM gap:   {ece_gap_before:+.6f}")
                print(f"      UASAC_R→UAMCM gap: {ece_gap_after:+.6f}")
                if abs(ece_gap_after) < abs(ece_gap_before):
                    print(f"      → Rescaling reduced UASAC-UAMCM ECE gap by {(1-abs(ece_gap_after)/abs(ece_gap_before))*100:.1f}%")

        # Q3: lam=1e-2 vs V5 (lam=1e-3 for AliCCP)
        print(f"\n  Q3: Note — V5 used lam=1e-3 for AliCCP, V6 uses lam=1e-2. Direct comparison requires V5 data.")

        # Q4: UMC vs UMC_WOR
        umc = next((r for r in ds_rows if r["name"] == "umc"), None)
        umc_wor = next((r for r in ds_rows if r["name"] == "umc_wor"), None)
        if umc and umc_wor:
            print(f"\n  Q4: UMC vs UMC_WOR (rescaling effect):")
            print(f"      umc      ECE={umc['ece']:.6f}  LL={umc['logloss']:.6f}  AUC={umc['auc']:.6f}  PCOC={umc['pcoc']:.6f}")
            print(f"      umc_wor  ECE={umc_wor['ece']:.6f}  LL={umc_wor['logloss']:.6f}  AUC={umc_wor['auc']:.6f}  PCOC={umc_wor['pcoc']:.6f}")
            ece_diff = umc_wor["ece"] - umc["ece"]
            ll_diff = umc_wor["logloss"] - umc["logloss"]
            print(f"      Rescaling effect: ECE {ece_diff:+.6f} ({ece_diff/umc_wor['ece']*100:+.1f}%)  LL {ll_diff:+.6f}")

        # Also check UAMCM vs UAMCM_WOR
        uamcm_wor = next((r for r in ds_rows if r["name"] == "uamcm_wor"), None)
        if uamcm and uamcm_wor:
            print(f"\n  Q4b: UAMCM vs UAMCM_WOR (rescaling effect with u-conditioning):")
            print(f"      uamcm      ECE={uamcm['ece']:.6f}  LL={uamcm['logloss']:.6f}  AUC={uamcm['auc']:.6f}  PCOC={uamcm['pcoc']:.6f}")
            print(f"      uamcm_wor  ECE={uamcm_wor['ece']:.6f}  LL={uamcm_wor['logloss']:.6f}  AUC={uamcm_wor['auc']:.6f}  PCOC={uamcm_wor['pcoc']:.6f}")
            ece_diff = uamcm_wor["ece"] - uamcm["ece"]
            print(f"      Rescaling effect: ECE {ece_diff:+.6f} ({ece_diff/uamcm_wor['ece']*100:+.1f}%)")


def scl_verification(rows):
    """Verify all neural methods used correct scl_lam and scl_beta."""
    print(f"\n{'='*90}")
    print(f"  SCL PARAMETER VERIFICATION")
    print(f"{'='*90}")

    neural_methods = [m for m in METHODS_ORDER if not m.startswith("sta_")]
    all_ok = True
    for dataset in DATASETS:
        for method in neural_methods:
            r = next((r for r in rows if r["dataset"] == dataset and r["name"] == method), None)
            if r is None:
                continue
            lam = r.get("scl_lam")
            beta = r.get("scl_beta")
            ok = (lam == 0.01 and beta == 0.95)
            status = "OK" if ok else "MISMATCH"
            if not ok:
                all_ok = False
                print(f"  {status}  {dataset}/{method}  lam={lam}  beta={beta}")

    if all_ok:
        print(f"  ALL NEURAL METHODS: scl_lam=0.01, scl_beta=0.95 — VERIFIED ✓")


def main():
    print("=" * 90)
    print("  V6 PHASE 1 — COMPREHENSIVE DATA ANALYSIS")
    print("=" * 90)

    rows = build_unified_summary()
    print(f"\n  Total experiments extracted: {len(rows)}")

    # Write unified CSV
    csv_path = os.path.join(OUT_DIR, "summary_v6_phase1_unified.csv")
    if rows:
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Unified CSV: {csv_path}")

    # SCL verification
    scl_verification(rows)

    # Rankings
    for dataset in DATASETS:
        print_ranking_table(rows, dataset, "ece", ascending=True)
        print_ece_best_ranking(rows, dataset)
        print_ranking_table(rows, dataset, "logloss", ascending=True)

    # PCOC-ECE correlation
    print(f"\n{'='*90}")
    print(f"  PCOC-ECE CORRELATION ANALYSIS")
    print(f"{'='*90}")
    for dataset in DATASETS:
        r = pcoc_ece_correlation(rows, dataset)
        if r is not None:
            print(f"  {dataset.upper()}: Pearson r(|PCOC-1|, ECE) = {r:.6f}")

    # Cross-dataset consistency
    cross_dataset_consistency(rows)

    # Key questions
    key_questions_analysis(rows)

    # Summary for paper direction
    print(f"\n{'='*90}")
    print(f"  PHASE 2 EXPERIMENT GROUP RECOMMENDATIONS")
    print(f"{'='*90}")

    for dataset in DATASETS:
        ds = [r for r in rows if r["dataset"] == dataset]
        uamcm = next((r for r in ds if r["name"] == "uamcm"), None)
        umc = next((r for r in ds if r["name"] == "umc"), None)
        uasac_r = next((r for r in ds if r["name"] == "uasac_r_K3"), None)

        if uamcm and umc:
            ece_sorted = sorted([r for r in ds if "ece" in r], key=lambda r: r["ece"])
            uamcm_rank = next(i for i, r in enumerate(ece_sorted, 1) if r["name"] == "uamcm")
            umc_rank = next(i for i, r in enumerate(ece_sorted, 1) if r["name"] == "umc")
            print(f"\n  [{dataset.upper()}]")
            print(f"  UAMCM ECE rank: #{uamcm_rank}  UMC ECE rank: #{umc_rank}")
            if uamcm_rank <= 3:
                print(f"  → Group A (Lambda sensitivity): RECOMMENDED — UAMCM is competitive")
                print(f"  → Group D (Multi-seed): RECOMMENDED — need to verify stability")
            if uasac_r:
                uasac_r_rank = next(i for i, r in enumerate(ece_sorted, 1) if r["name"] == "uasac_r_K3")
                print(f"  UASAC_R ECE rank: #{uasac_r_rank}")
                if uasac_r_rank <= 5:
                    print(f"  → Group B (UASAC_R arch search): RECOMMENDED — competitive")
                else:
                    print(f"  → Group B (UASAC_R arch search): LOW PRIORITY — not competitive")


if __name__ == "__main__":
    main()
