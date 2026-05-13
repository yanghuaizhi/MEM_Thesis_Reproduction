#!/usr/bin/env python
"""V6 experiment runner — Phase 1: Fair baseline comparison.

Usage:
  # Phase 1: Fair baselines on AliCCP (GPU 0)
  python run_v6.py --gpu 0 --dataset aliccp

  # Phase 1: Fair baselines on Avazu (GPU 1)
  python run_v6.py --gpu 1 --dataset avazu

  # Custom output directory:
  python run_v6.py --gpu 0 --dataset aliccp --out-dir /root/shared-nvme/PAPER/ckpt/v6_phase1
"""

import os
import sys
import csv
import re
import argparse
import torch
from datetime import datetime
from io import StringIO

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "aliccp": {
        "data_name": "aliccp",
        "field_index": 0,
    },
    "avazu": {
        "data_name": "avazu",
        "field_index": 2,
    },
}

# ---------------------------------------------------------------------------
# Backbone config (shared by all methods)
# ---------------------------------------------------------------------------
BACKBONE_COMMON = {
    "data_root": "/root/shared-nvme/PAPER/dataset",
    "filepath": "/root/shared-nvme/PAPER/ckpt",
    "model_name": "packed_deepfm",
    "batch_size": 1024 * 32,
    "dropout": 0.1,
    "init_std": 1e-4,
    "lr": 5e-4,
    "l2_reg": 1e-5,
    "seed": 1024,
    "num_estimators": 16,
    "alpha": 1.0,
    "gamma": 1,
}

# ---------------------------------------------------------------------------
# Unified neural calibration config (V6: lam=1e-2, beta=0.95)
# ---------------------------------------------------------------------------
NEU_COMMON = {
    "lr_calib": 1e-3,
    "epochs_calib": 20,
    "batch_size_calib": 1024 * 64,
    "calib_early_stop": True,
    "calib_patience": 5,
    "calib_min_delta": 1e-5,
    "calib_restore_best": True,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "uncertainty_bin_eval": True,
    "uncertainty_bin_num_bins": 20,
    "uncertainty_bin_ece_M": 100,
    "calib_log_every": 200,
    "ece_M": 100,
    "scl_lam": 1e-2,
    "scl_beta": 0.95,
}

# ---------------------------------------------------------------------------
# Statistical calibration config
# ---------------------------------------------------------------------------
STA_COMMON = {
    "batch_size_calib": 1024 * 64,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "uncertainty_bin_eval": True,
    "uncertainty_bin_num_bins": 20,
    "uncertainty_bin_ece_M": 100,
    "ece_M": 100,
}

# ---------------------------------------------------------------------------
# Avazu OOM-aware batch sizes: UMC/UAMCM/UASAC series need smaller batches
# ---------------------------------------------------------------------------
AVAZU_SMALL_BATCH_METHODS = {
    "umc_wor", "umc", "uamcm_wor", "uamcm", "uamcm_no_u_rs",
    "uamcm_phase4", "uasac", "uasac_r",
}
AVAZU_SMALL_BATCH_SIZE = 1024 * 16


# ===========================================================================
# Phase 1: Fair baseline comparison (14 methods)
# ===========================================================================
PHASE1_NEU = [
    {"name": "neu",        "method": "neu"},
    {"name": "desc",       "method": "desc"},
    {"name": "sbcr",       "method": "sbcr"},
    {"name": "umnn",       "method": "umnn"},
    {"name": "umc_wor",    "method": "umc_wor"},
    {"name": "umc",        "method": "umc"},
    {"name": "uamcm_wor",  "method": "uamcm_wor"},
    {"name": "uamcm",      "method": "uamcm"},
    {"name": "uasac_K3",   "method": "uasac",
     "num_experts": 3, "expert_dim": 16, "router_type": "mlp",
     "router_hidden": 64, "temperature": 1.0, "div_lambda": 0.0},
    {"name": "uasac_r_K3", "method": "uasac_r",
     "num_experts": 3, "expert_dim": 16, "router_type": "mlp",
     "router_hidden": 64, "temperature": 1.0, "div_lambda": 0.0},
    {"name": "uamcm_phase4", "method": "uamcm_phase4",
     "ra_weighted_bce": True, "ra_weight_c": 1.0, "ra_weight_k": 1.0,
     "distill_lambda": 0.1, "distill_t": 1.0},
]

PHASE1_STA = [
    {"name": "sta_ir",    "method": "ir"},
    {"name": "sta_hb",    "method": "hb"},
    {"name": "sta_platt", "method": "platt"},
]

PHASE1_ALL = PHASE1_NEU + PHASE1_STA


# ===========================================================================
# Utilities
# ===========================================================================
class TeeOutput:
    """Write to both original stdout and a StringIO capture."""

    def __init__(self, orig, cap):
        self.orig = orig
        self.cap = cap

    def write(self, s):
        self.orig.write(s)
        self.orig.flush()
        self.cap.write(s)

    def flush(self):
        self.orig.flush()
        self.cap.flush()


def extract_metrics(log_content, tag="calibrated"):
    """Extract metrics following an exact metrics_tag=<tag> line."""
    metrics = {}
    found_tag = False
    for line in log_content.split("\n"):
        if line.strip() == f"metrics_tag={tag}":
            found_tag = True
            continue
        if found_tag and "test_" in line:
            for m in re.finditer(r"test_(\w+)\s*=\s*([\d.eE+-]+)", line):
                metrics[m.group(1)] = m.group(2)
            break
    return metrics


def build_config(dataset, exp):
    """Build a complete config_update dict for one experiment."""
    ds_conf = DATASET_CONFIG[dataset]
    is_statistical = exp.get("method") in ("ir", "hb", "platt")

    config_update = {}
    config_update.update(BACKBONE_COMMON)
    config_update.update(ds_conf)
    config_update.update(STA_COMMON if is_statistical else NEU_COMMON)

    # Method-specific params
    for k, v in exp.items():
        if k != "name":
            config_update[k] = v

    # Avazu OOM-aware batch size
    method = exp.get("method", "")
    if dataset == "avazu" and method in AVAZU_SMALL_BATCH_METHODS:
        config_update["batch_size_calib"] = AVAZU_SMALL_BATCH_SIZE

    return config_update, is_statistical


def run_single_experiment(trial_fn, config_update, name, out_dir, exp_idx, total, gpu_id, dataset):
    """Run one experiment with output capture and metric extraction."""
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    method = config_update.get("method", "?")
    log_path = os.path.join(log_dir, f"{name}.log")

    print(f"\n{'='*60}")
    print(f"  [{exp_idx}/{total}] {name}  method={method}")
    print(f"  GPU={gpu_id}  dataset={dataset}")
    print(f"  scl_lam={config_update.get('scl_lam', 'N/A')}  scl_beta={config_update.get('scl_beta', 'N/A')}")
    print(f"{'='*60}\n")

    old_stdout = sys.stdout
    captured = StringIO()
    sys.stdout = TeeOutput(old_stdout, captured)

    try:
        trial_fn(config_update)
    except Exception as e:
        import traceback
        print(f"EXPERIMENT_FAILED name={name} error={e}")
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout

    log_content = captured.getvalue()
    with open(log_path, "w") as f:
        f.write(log_content)

    # Extract metrics
    row = {"name": name, "dataset": dataset, "method": method}
    row["seed"] = config_update.get("seed", 1024)
    row["scl_lam"] = config_update.get("scl_lam", "N/A")

    for tag in ("calibrated", "ece_best_calibrated"):
        m = extract_metrics(log_content, tag)
        prefix = "" if tag == "calibrated" else "ece_best_"
        for k, v in m.items():
            row[f"{prefix}{k}"] = v

    print(f"  FINISHED: {name}")
    for k in ("ece", "logloss", "auc", "pcoc"):
        print(f"    {k} = {row.get(k, 'N/A')}")
        print(f"    ece_best_{k} = {row.get(f'ece_best_{k}', 'N/A')}")

    # Free GPU memory
    torch.cuda.empty_cache()

    return row


def main():
    parser = argparse.ArgumentParser(description="V6 Phase 1 Experiment Runner")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["aliccp", "avazu"], help="Dataset",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: /root/shared-nvme/PAPER/ckpt/v6_phase1)",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated experiment names to run (default: all Phase 1)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or "/root/shared-nvme/PAPER/ckpt/v6_phase1"
    os.makedirs(out_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Import trial functions
    from train_neu_ali import trial as trial_neu_ali
    from train_neu_avazu import trial as trial_neu_avazu
    from train_sta_ali import trial as trial_sta_ali
    from train_sta_avazu import trial as trial_sta_avazu

    trial_neu = trial_neu_ali if args.dataset == "aliccp" else trial_neu_avazu
    trial_sta = trial_sta_ali if args.dataset == "aliccp" else trial_sta_avazu

    if args.methods:
        method_set = set(args.methods.split(","))
        experiments = [e for e in PHASE1_ALL if e["name"] in method_set]
    else:
        experiments = PHASE1_ALL
    total = len(experiments)

    print(f"V6 Phase 1 Experiment Runner")
    print(f"  GPU: {args.gpu}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Experiments: {total}")
    print(f"  Output: {out_dir}")
    print(f"  scl_lam: {NEU_COMMON['scl_lam']}  scl_beta: {NEU_COMMON['scl_beta']}")
    print(f"  Start: {datetime.now().isoformat()}")

    summary_rows = []
    for idx, exp in enumerate(experiments, 1):
        name = f"{args.dataset}_{exp['name']}"
        config_update, is_sta = build_config(args.dataset, exp)

        # Ubins save path
        config_update["uncertainty_bin_save_path"] = os.path.join(
            out_dir, f"{name}_ubins.csv"
        )

        trial_fn = trial_sta if is_sta else trial_neu
        row = run_single_experiment(
            trial_fn, config_update, name, out_dir,
            idx, total, args.gpu, args.dataset,
        )
        row["group"] = "phase1"
        summary_rows.append(row)

    # Write summary CSV
    suffix = "_partial" if args.methods else ""
    summary_path = os.path.join(out_dir, f"summary_{args.dataset}_phase1{suffix}.csv")
    if summary_rows:
        all_keys = []
        for r in summary_rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print(f"  V6 PHASE 1 COMPLETE - {args.dataset}")
    print(f"  Summary: {summary_path}")
    print(f"  Experiments: {len(summary_rows)}")
    print(f"  End: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
