#!/usr/bin/env python
"""V7 Supplement experiment runner — Multi-seed supplement for all calibration methods.

Supplements V7 with:
  - 8 methods × 3 seeds × 2 datasets = 48 neural experiments (full-seed methods)
  - 3 methods × 1 seed(1024) × 2 datasets = 6 neural experiments (V7 has 2024/3024)
  - 3 stat methods × 1 run × 2 datasets = 6 experiments (deterministic)
  Total: 60 experiments

GPU load balancing (~12h wall clock):
  GPU0 (~680 min): AliCCP s1024 (11 neu) + AliCCP s2024 (8 neu) + AliCCP stat (3)
  GPU1 (~721 min): AliCCP s3024 (8 neu) + ALL Avazu (27 neu + 3 stat)

Usage:
  python run_v7_supplement.py --gpu 0          # Run GPU0 queue
  python run_v7_supplement.py --gpu 1          # Run GPU1 queue
  python run_v7_supplement.py --gpu 0 --smoke  # Smoke test (1 experiment)
  python run_v7_supplement.py --merge          # Merge gpu0+gpu1 → unified
  python run_v7_supplement.py --merge-v7       # Merge V7+V7_supp → mean±std

Output: ckpt/v7_supp/ (separate from V7 to avoid contamination)
"""

import os
import sys
import csv
import re
import argparse
import numpy as np
import torch
from datetime import datetime

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

# ---------------------------------------------------------------------------
# Dataset-specific configuration (same as run_v7.py)
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
# Backbone config (shared, seed=1024 is required for checkpoint path)
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
    "seed": 1024,  # MUST stay 1024 for backbone checkpoint loading
    "num_estimators": 16,
    "alpha": 1.0,
    "gamma": 1,
}

# ---------------------------------------------------------------------------
# Neural calibration config (same as run_v7.py)
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
# Statistical calibration config (no neural-specific keys like lr_calib etc.)
# ---------------------------------------------------------------------------
STAT_COMMON = {
    "batch_size_calib": 1024 * 64,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "ece_M": 100,
    "uncertainty_bin_eval": True,
    "uncertainty_bin_num_bins": 20,
    "uncertainty_bin_ece_M": 100,
}

# ---------------------------------------------------------------------------
# Avazu OOM-aware batch sizes (same as run_v7.py)
# ---------------------------------------------------------------------------
AVAZU_SMALL_BATCH_METHODS = {
    "umc_wor", "umc", "uamcm_wor", "uamcm", "uamcm_no_u_rs",
    "uamcm_phase4", "uasac", "uasac_r",
}
AVAZU_SMALL_BATCH_SIZE = 1024 * 16

# ===========================================================================
# Experiment definitions
# ===========================================================================

# Neural methods needing all 3 seeds (no V7 multi-seed data exists)
_FULL_SEED = [
    ("neu", "neu", {}),
    ("desc", "desc", {}),
    ("sbcr", "sbcr", {}),
    ("umnn", "umnn", {}),
    ("umc_wor", "umc_wor", {}),
    ("uamcm_wor", "uamcm_wor", {}),
    ("uasac", "uasac", {
        "num_experts": 3, "expert_dim": 16,
        "router_type": "mlp", "router_hidden": 64,
        "temperature": 1.0, "div_lambda": 0.0, "router_mode": "full",
    }),
    ("uamcm_phase4", "uamcm_phase4", {
        "ra_weighted_bce": True, "ra_weight_c": 1.0, "ra_weight_k": 1.0,
        "distill_lambda": 0.1, "distill_t": 1.0,
    }),
]

# Neural methods needing only seed=1024 (V7 already has seeds 2024/3024)
_PARTIAL_SEED = [
    ("umc", "umc", {}),
    ("uamcm", "uamcm", {}),
    ("uasac_r_K3", "uasac_r", {
        "num_experts": 3, "expert_dim": 16,
        "router_type": "mlp", "router_hidden": 64,
        "temperature": 1.0, "div_lambda": 0.0, "router_mode": "full",
    }),
]

# Stat methods (deterministic, single run)
_STAT_METHODS = ["ir", "hb", "platt"]

# ---------------------------------------------------------------------------
# Generate all experiments programmatically
# ---------------------------------------------------------------------------
NEU_EXPERIMENTS = []
for _display, _method, _extra in _FULL_SEED:
    for _seed in [1024, 2024, 3024]:
        _exp = {
            "name": f"{_display}_s{_seed}",
            "method": _method,
            "calib_seed": _seed,
            "trial_type": "neu",
        }
        _exp.update(_extra)
        NEU_EXPERIMENTS.append(_exp)

for _display, _method, _extra in _PARTIAL_SEED:
    _exp = {
        "name": f"{_display}_s1024",
        "method": _method,
        "calib_seed": 1024,
        "trial_type": "neu",
    }
    _exp.update(_extra)
    NEU_EXPERIMENTS.append(_exp)

STA_EXPERIMENTS = [
    {"name": m, "method": m, "trial_type": "sta"}
    for m in _STAT_METHODS
]

ALL_EXPERIMENTS = {e["name"]: e for e in NEU_EXPERIMENTS + STA_EXPERIMENTS}

# ---------------------------------------------------------------------------
# GPU queue assignment (load-balanced)
#   GPU0 (~680 min): AliCCP s1024 (11 neu) + AliCCP s2024 (8 neu) + AliCCP stat (3)
#     = 11×35 + 8×35 + 3×5 = 680 min
#   GPU1 (~727 min): AliCCP s3024 (8 neu) + ALL Avazu (27 neu + 3 stat)
#     = 8×35 + 27×16 + 3×5 = 727 min
# ---------------------------------------------------------------------------
GPU_QUEUES = {
    0: (
        # AliCCP seed=1024: 8 full-seed + 3 partial-seed = 11 neural
        [("aliccp", f"{n}_s1024") for n, _, _ in _FULL_SEED]
        + [("aliccp", f"{n}_s1024") for n, _, _ in _PARTIAL_SEED]
        # AliCCP seed=2024: 8 full-seed
        + [("aliccp", f"{n}_s2024") for n, _, _ in _FULL_SEED]
        # AliCCP stat: 3
        + [("aliccp", m) for m in _STAT_METHODS]
    ),
    1: (
        # AliCCP seed=3024: 8 full-seed
        [("aliccp", f"{n}_s3024") for n, _, _ in _FULL_SEED]
        # Avazu neural: 8×3 full-seed + 3 partial-seed = 27
        + [("avazu", f"{n}_s{s}") for n, _, _ in _FULL_SEED for s in [1024, 2024, 3024]]
        + [("avazu", f"{n}_s1024") for n, _, _ in _PARTIAL_SEED]
        # Avazu stat: 3
        + [("avazu", m) for m in _STAT_METHODS]
    ),
}


# ===========================================================================
# Utilities (reused from run_v7.py)
# ===========================================================================

class TeeOutput:
    """Write to both original stdout and an open log file (real-time flush)."""

    def __init__(self, orig, log_file):
        self.orig = orig
        self.log_file = log_file

    def write(self, s):
        self.orig.write(s)
        self.orig.flush()
        self.log_file.write(s)
        self.log_file.flush()

    def flush(self):
        self.orig.flush()
        self.log_file.flush()


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


def extract_epoch_trajectory(log_content):
    """Extract per-epoch training trajectory from log for learning-curve figures."""
    rows = []
    lines = log_content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_epoch = re.search(
            r"calib_epoch_end epoch=(\d+)/\d+ epoch_loss=([\d.eE+-]+)"
            r".*best_epoch_loss=([\d.eE+-]+).*best_epoch=(\d+)",
            line,
        )
        if m_epoch:
            epoch = int(m_epoch.group(1))
            train_loss = float(m_epoch.group(2))
            best_loss = float(m_epoch.group(3))
            best_epoch = int(m_epoch.group(4))
            test_metrics = {}
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].strip() == "metrics_tag=calibrated_fast":
                    for k in range(j + 1, min(j + 5, len(lines))):
                        if "test_" in lines[k]:
                            for mm in re.finditer(
                                r"test_(\w+)\s*=\s*([\d.eE+-]+)", lines[k]
                            ):
                                test_metrics[mm.group(1)] = float(mm.group(2))
                            break
                    break
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "best_train_loss": best_loss,
                "best_epoch": best_epoch,
                "test_logloss": test_metrics.get("logloss", ""),
                "test_auc": test_metrics.get("auc", ""),
                "test_ece": test_metrics.get("ece", ""),
                "test_pcoc": test_metrics.get("pcoc", ""),
            }
            rows.append(row)
        i += 1
    return rows


def build_config_neu(dataset, exp):
    """Build a complete config_update dict for a neural experiment."""
    ds_conf = DATASET_CONFIG[dataset]
    config_update = {}
    config_update.update(BACKBONE_COMMON)
    config_update.update(ds_conf)
    config_update.update(NEU_COMMON)

    for k, v in exp.items():
        if k not in ("name", "trial_type"):
            config_update[k] = v

    method = exp.get("method", "")
    if dataset == "avazu" and method in AVAZU_SMALL_BATCH_METHODS:
        config_update["batch_size_calib"] = AVAZU_SMALL_BATCH_SIZE

    return config_update


def build_config_sta(dataset, exp):
    """Build a complete config_update dict for a statistical experiment."""
    ds_conf = DATASET_CONFIG[dataset]
    config_update = {}
    config_update.update(BACKBONE_COMMON)
    config_update.update(ds_conf)
    config_update.update(STAT_COMMON)
    config_update["method"] = exp["method"]
    return config_update


def run_single_experiment(trial_fn, config_update, name, out_dir, exp_idx, total,
                          gpu_id, dataset, trial_type):
    """Run one experiment with output capture and metric extraction."""
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    method = config_update.get("method", "?")
    calib_seed = config_update.get("calib_seed", "N/A")
    log_path = os.path.join(log_dir, f"{name}.log")

    print(f"\n{'='*60}")
    print(f"  [{exp_idx}/{total}] {name}  method={method}  type={trial_type}")
    print(f"  GPU={gpu_id}  dataset={dataset}  calib_seed={calib_seed}")
    if trial_type == "neu":
        print(f"  router_mode={config_update.get('router_mode', 'N/A')}")
        print(f"  div_lambda={config_update.get('div_lambda', 'N/A')}")
        print(f"  scl_lam={config_update.get('scl_lam', 'N/A')}  scl_beta={config_update.get('scl_beta', 'N/A')}")
        print(f"  batch_size_calib={config_update.get('batch_size_calib', 'N/A')}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    old_stdout = sys.stdout
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = TeeOutput(old_stdout, log_file)

    try:
        trial_fn(config_update)
    except Exception as e:
        import traceback
        print(f"EXPERIMENT_FAILED name={name} error={e}")
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout
        log_file.close()

    with open(log_path, "r") as f:
        log_content = f.read()

    row = {
        "name": name,
        "dataset": dataset,
        "method": method,
        "trial_type": trial_type,
        "calib_seed": calib_seed,
    }

    if trial_type == "neu":
        row["router_mode"] = config_update.get("router_mode", "N/A")
        row["div_lambda"] = config_update.get("div_lambda", "N/A")
        row["scl_lam"] = config_update.get("scl_lam", "N/A")

    # Extract metrics (stat methods only produce "calibrated", not "ece_best_calibrated")
    for tag in ("calibrated", "ece_best_calibrated"):
        m = extract_metrics(log_content, tag)
        prefix = "" if tag == "calibrated" else "ece_best_"
        for k, v in m.items():
            row[f"{prefix}{k}"] = v

    # Save per-epoch trajectory CSV (neural only)
    if trial_type == "neu":
        epoch_rows = extract_epoch_trajectory(log_content)
        if epoch_rows:
            epoch_path = os.path.join(log_dir, f"{name}_epochs.csv")
            _write_csv(epoch_rows, epoch_path)

    print(f"  FINISHED: {name}  ({datetime.now().isoformat()})")
    for k in ("ece", "logloss", "auc", "pcoc"):
        print(f"    loss_best_{k} = {row.get(k, 'N/A')}")
        if trial_type == "neu":
            print(f"    ece_best_{k}  = {row.get(f'ece_best_{k}', 'N/A')}")

    torch.cuda.empty_cache()

    return row


def _write_csv(rows, path):
    if not rows:
        return
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def _merge_results(out_dir, silent=False):
    """Merge gpu0 and gpu1 summary CSVs into a unified file."""
    paths = [
        os.path.join(out_dir, f"summary_v7_supp_gpu{g}.csv")
        for g in [0, 1]
    ]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        if not silent:
            print(f"Cannot merge: missing files: {missing}")
        return

    all_rows = []
    all_keys = []
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                for k in row:
                    if k not in all_keys:
                        all_keys.append(k)

    unified_path = os.path.join(out_dir, "summary_v7_supp_unified.csv")
    with open(unified_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Merged {len(all_rows)} experiments -> {unified_path}")


def _merge_v7(supp_dir):
    """Merge V7 and V7_supp into summary_v7_full_meanstd.csv with mean +/- std."""
    v7_path = "/root/shared-nvme/PAPER/ckpt/v7/summary_v7_unified.csv"
    supp_path = os.path.join(supp_dir, "summary_v7_supp_unified.csv")

    all_rows = []
    for path in [v7_path, supp_path]:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            return
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    def get_base_method(name, dataset):
        """Extract base method name: strip dataset prefix, then _s{seed} suffix."""
        prefix = f"{dataset}_"
        if name.startswith(prefix):
            name = name[len(prefix):]
        return re.sub(r"_s\d+$", "", name)

    # Group by (dataset, base_method)
    from collections import defaultdict
    groups = defaultdict(list)
    for row in all_rows:
        dataset = row["dataset"]
        base_method = get_base_method(row["name"], dataset)
        groups[(dataset, base_method)].append(row)

    METRICS = ["logloss", "ece", "auc", "gauc", "pcoc", "rce", "mfece", "mfrce"]

    result_rows = []
    for (dataset, base_method) in sorted(groups.keys()):
        rows = groups[(dataset, base_method)]
        result = {
            "dataset": dataset,
            "method": base_method,
            "n_seeds": len(rows),
        }
        for metric in METRICS:
            vals = []
            for r in rows:
                v = r.get(metric, "")
                if v and v != "N/A":
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals)) if len(vals) > 1 else 0.0
                result[f"{metric}_mean"] = f"{mean:.6f}"
                result[f"{metric}_std"] = f"{std:.6f}"
                result[f"{metric}"] = f"{mean:.6f}\u00b1{std:.6f}"
            else:
                result[f"{metric}_mean"] = ""
                result[f"{metric}_std"] = ""
                result[f"{metric}"] = ""
        result_rows.append(result)

    out_path = os.path.join(supp_dir, "summary_v7_full_meanstd.csv")
    _write_csv(result_rows, out_path)
    print(f"\n  Merged {len(all_rows)} experiment rows -> {len(result_rows)} method x dataset rows")
    print(f"  Output: {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="V7 Supplement Experiment Runner")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index (0 or 1). Required unless using --merge/--merge-v7.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: run only the first experiment from the GPU queue.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: /root/shared-nvme/PAPER/ckpt/v7_supp)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge gpu0.csv and gpu1.csv into summary_v7_supp_unified.csv.",
    )
    parser.add_argument(
        "--merge-v7", action="store_true",
        help="Merge V7 + V7_supp -> summary_v7_full_meanstd.csv (mean +/- std).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or "/root/shared-nvme/PAPER/ckpt/v7_supp"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Merge-only modes
    # ------------------------------------------------------------------
    if args.merge_v7:
        _merge_v7(out_dir)
        return

    if args.merge and args.gpu is None:
        _merge_results(out_dir)
        return

    if args.gpu is None:
        parser.error("--gpu is required unless using --merge or --merge-v7.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Import trial functions (after setting CUDA_VISIBLE_DEVICES)
    from train_neu_ali import trial as trial_neu_ali
    from train_neu_avazu import trial as trial_neu_avazu
    from train_sta_ali import trial as trial_sta_ali
    from train_sta_avazu import trial as trial_sta_avazu

    TRIAL_FNS = {
        ("aliccp", "neu"): trial_neu_ali,
        ("avazu", "neu"): trial_neu_avazu,
        ("aliccp", "sta"): trial_sta_ali,
        ("avazu", "sta"): trial_sta_avazu,
    }

    # ------------------------------------------------------------------
    # Build experiment queue
    # ------------------------------------------------------------------
    queue = list(GPU_QUEUES[args.gpu])
    if args.smoke:
        queue = queue[:1]

    total = len(queue)
    print(f"\nV7 Supplement Experiment Runner")
    print(f"  GPU: {args.gpu}")
    print(f"  Experiments: {total}")
    print(f"  Output: {out_dir}")
    print(f"  Smoke: {args.smoke}")
    print(f"  Start: {datetime.now().isoformat()}")

    # ------------------------------------------------------------------
    # Run queue
    # ------------------------------------------------------------------
    summary_rows = []
    for idx, (dataset, exp_name) in enumerate(queue, 1):
        if exp_name not in ALL_EXPERIMENTS:
            print(f"  SKIP: {exp_name} not found in ALL_EXPERIMENTS")
            continue

        exp = ALL_EXPERIMENTS[exp_name]
        trial_type = exp["trial_type"]

        # Build config (different builder for neu vs sta)
        if trial_type == "neu":
            config_update = build_config_neu(dataset, exp)
        else:
            config_update = build_config_sta(dataset, exp)

        # Ubins save path
        run_name = f"{dataset}_{exp_name}"
        config_update["uncertainty_bin_save_path"] = os.path.join(
            out_dir, f"{run_name}_ubins.csv"
        )

        # Select trial function
        trial_fn = TRIAL_FNS[(dataset, trial_type)]

        row = run_single_experiment(
            trial_fn, config_update, run_name, out_dir,
            idx, total, args.gpu, dataset, trial_type,
        )
        summary_rows.append(row)

    # ------------------------------------------------------------------
    # Write per-GPU summary CSV
    # ------------------------------------------------------------------
    suffix = "_smoke" if args.smoke else ""
    summary_path = os.path.join(out_dir, f"summary_v7_supp_gpu{args.gpu}{suffix}.csv")
    _write_csv(summary_rows, summary_path)

    print(f"\n{'='*60}")
    print(f"  V7 Supplement GPU {args.gpu} COMPLETE")
    print(f"  Summary: {summary_path}")
    print(f"  Experiments: {len(summary_rows)}")
    print(f"  End: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    # Attempt merge if both CSVs exist (non-smoke mode)
    if not args.smoke:
        _merge_results(out_dir, silent=True)


if __name__ == "__main__":
    main()
