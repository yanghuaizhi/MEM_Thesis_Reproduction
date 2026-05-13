#!/usr/bin/env python
"""V7 experiment runner — One-shot parallel execution on 2 GPUs.

Usage:
  # Run GPU 0's queue (515 min, ~8.6h):
  python run_v7.py --gpu 0

  # Run GPU 1's queue:
  python run_v7.py --gpu 1

  # Smoke test — single experiment (Avazu + random router, ~33min):
  python run_v7.py --gpu 0 --dataset avazu --methods uasac_r_random

  # Launch both GPUs at once (run in shell):
  for g in 0 1; do
    nohup python run_v7.py --gpu $g > /root/shared-nvme/PAPER/ckpt/v7/gpu${g}.log 2>&1 &
  done

  # Merge results after both GPUs complete:
  python run_v7.py --merge
"""

import os
import sys
import csv
import re
import argparse
import torch
from datetime import datetime

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

# ---------------------------------------------------------------------------
# Dataset-specific configuration (same as V6)
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
# Neural calibration config (V6: lam=1e-2, beta=0.95)
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
# Avazu OOM-aware batch sizes
# ---------------------------------------------------------------------------
AVAZU_SMALL_BATCH_METHODS = {
    "umc_wor", "umc", "uamcm_wor", "uamcm", "uamcm_no_u_rs",
    "uasac", "uasac_r",
}
AVAZU_SMALL_BATCH_SIZE = 1024 * 16

# ===========================================================================
# Experiment definitions
# ===========================================================================

# Shared UASAC_R base params
_UASAC_R_BASE = {
    "method": "uasac_r",
    "num_experts": 3,
    "expert_dim": 16,
    "router_type": "mlp",
    "router_hidden": 64,
    "temperature": 1.0,
}

# ---------------------------------------------------------------------------
# Group A: Multi-seed robustness (12 experiments, seeds 2024 & 3024)
# ---------------------------------------------------------------------------
GROUP_A = []
for _method_name, _method_key, _extra in [
    ("uasac_r_K3", "uasac_r", {
        "num_experts": 3, "expert_dim": 16,
        "router_type": "mlp", "router_hidden": 64,
        "temperature": 1.0, "div_lambda": 0.0, "router_mode": "full",
    }),
    ("uamcm", "uamcm", {}),
    ("umc", "umc", {}),
]:
    for _seed in [2024, 3024]:
        _exp = {
            "name": f"{_method_name}_s{_seed}",
            "method": _method_key,
            "calib_seed": _seed,
        }
        _exp.update(_extra)
        GROUP_A.append(_exp)

# ---------------------------------------------------------------------------
# Group B: UASAC improvements (6 experiments, calib_seed=1024)
# ---------------------------------------------------------------------------
GROUP_B = [
    {
        "name": "uasac_r_div001",
        "method": "uasac_r",
        "num_experts": 3, "expert_dim": 16,
        "router_type": "mlp", "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.01,
        "router_mode": "full",
    },
    {
        "name": "uasac_r_uonly",
        "method": "uasac_r",
        "num_experts": 3, "expert_dim": 16,
        "router_type": "mlp", "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.0,
        "router_mode": "u_only",
    },
    {
        "name": "uasac_r_div001_uonly",
        "method": "uasac_r",
        "num_experts": 3, "expert_dim": 16,
        "router_type": "mlp", "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.01,
        "router_mode": "u_only",
    },
]

# ---------------------------------------------------------------------------
# Group C: Ablation — random router (2 experiments, calib_seed=1024)
# ---------------------------------------------------------------------------
GROUP_C = [
    {
        "name": "uasac_r_random",
        "method": "uasac_r",
        "num_experts": 3, "expert_dim": 16,
        "router_type": "mlp", "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.0,
        "router_mode": "random",
    },
]

# ---------------------------------------------------------------------------
# Group D: New innovation validation — single seed (s1024), 3 new methods
# V1-V6: uamcm_no_u_rs, uamcm_dascl, uamcm_no_u_rs_dascl × {aliccp, avazu}
# ---------------------------------------------------------------------------
_DASCL_DEFAULT_LAM = 0.01

GROUP_D = [
    # V1-V2: UAMCM-D (decoupled: u only in integrand, not rescaling)
    {"name": "uamcm_no_u_rs", "method": "uamcm_no_u_rs", "calib_seed": 1024},
    # V4-V5: UAMCM + DA-SCL (density-aligned calibration loss)
    {"name": "uamcm_dascl", "method": "uamcm_dascl", "calib_seed": 1024,
     "dascl_lam": _DASCL_DEFAULT_LAM, "dascl_bins": 10},
    # V7-V8: UAMCM-D + DA-SCL (combined)
    {"name": "uamcm_no_u_rs_dascl", "method": "uamcm_no_u_rs_dascl", "calib_seed": 1024,
     "dascl_lam": _DASCL_DEFAULT_LAM, "dascl_bins": 10},
]

ALL_EXPERIMENTS = {e["name"]: e for e in GROUP_A + GROUP_B + GROUP_C + GROUP_D}

# ---------------------------------------------------------------------------
# GPU queue assignment (load-balanced: 5 AliCCP + 5 Avazu per GPU)
# Total per GPU: 5×70 + 5×33 = 515 min ≈ 8h35m
# ---------------------------------------------------------------------------
GPU_QUEUES = {
    0: [
        # A1, A5, A9: seed=2024 multi-seed (AliCCP)
        ("aliccp", "uasac_r_K3_s2024"),
        ("aliccp", "uamcm_s2024"),
        ("aliccp", "umc_s2024"),
        # B1, B3: div_lambda and u-only improvements (AliCCP)
        ("aliccp", "uasac_r_div001"),
        ("aliccp", "uasac_r_uonly"),
        # A3, A7, A11: seed=2024 multi-seed (Avazu)
        ("avazu",  "uasac_r_K3_s2024"),
        ("avazu",  "uamcm_s2024"),
        ("avazu",  "umc_s2024"),
        # B2, B4: div_lambda and u-only improvements (Avazu)
        ("avazu",  "uasac_r_div001"),
        ("avazu",  "uasac_r_uonly"),
    ],
    1: [
        # A2, A6, A10: seed=3024 multi-seed (AliCCP)
        ("aliccp", "uasac_r_K3_s3024"),
        ("aliccp", "uamcm_s3024"),
        ("aliccp", "umc_s3024"),
        # B5, C1: combined improvement + random ablation (AliCCP)
        ("aliccp", "uasac_r_div001_uonly"),
        ("aliccp", "uasac_r_random"),
        # A4, A8, A12: seed=3024 multi-seed (Avazu)
        ("avazu",  "uasac_r_K3_s3024"),
        ("avazu",  "uamcm_s3024"),
        ("avazu",  "umc_s3024"),
        # B6, C2: combined improvement + random ablation (Avazu)
        ("avazu",  "uasac_r_div001_uonly"),
        ("avazu",  "uasac_r_random"),
    ],
}

# ===========================================================================
# Utilities (reused from run_v6.py)
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
    """Extract per-epoch training trajectory from log for paper learning-curve figures.

    Parses:
      calib_epoch_end epoch=N/M epoch_loss=X.XX ...
      metrics_tag=calibrated_fast  (next test_* line)

    Returns list of dicts: {epoch, train_loss, test_logloss, test_auc, test_ece, test_pcoc}
    """
    rows = []
    lines = log_content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Parse train loss from epoch end line
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
            # Look ahead for metrics_tag=calibrated_fast and the metrics line
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


def build_config(dataset, exp):
    """Build a complete config_update dict for one experiment."""
    ds_conf = DATASET_CONFIG[dataset]
    config_update = {}
    config_update.update(BACKBONE_COMMON)
    config_update.update(ds_conf)
    config_update.update(NEU_COMMON)

    for k, v in exp.items():
        if k != "name":
            config_update[k] = v

    method = exp.get("method", "")
    if dataset == "avazu" and method in AVAZU_SMALL_BATCH_METHODS:
        config_update["batch_size_calib"] = AVAZU_SMALL_BATCH_SIZE

    return config_update


def run_single_experiment(trial_fn, config_update, name, out_dir, exp_idx, total, gpu_id, dataset):
    """Run one experiment with output capture and metric extraction."""
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    method = config_update.get("method", "?")
    router_mode = config_update.get("router_mode", "N/A")
    calib_seed = config_update.get("calib_seed", 1024)
    log_path = os.path.join(log_dir, f"{name}.log")

    print(f"\n{'='*60}")
    print(f"  [{exp_idx}/{total}] {name}  method={method}")
    print(f"  GPU={gpu_id}  dataset={dataset}")
    print(f"  router_mode={router_mode}  calib_seed={calib_seed}")
    print(f"  div_lambda={config_update.get('div_lambda', 'N/A')}")
    print(f"  scl_lam={config_update.get('scl_lam', 'N/A')}  scl_beta={config_update.get('scl_beta', 'N/A')}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    old_stdout = sys.stdout
    log_file = open(log_path, "w", buffering=1)  # line-buffered real-time write
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
        "calib_seed": calib_seed,
        "router_mode": router_mode,
        "div_lambda": config_update.get("div_lambda", "N/A"),
        "scl_lam": config_update.get("scl_lam", "N/A"),
    }

    for tag in ("calibrated", "ece_best_calibrated"):
        m = extract_metrics(log_content, tag)
        prefix = "" if tag == "calibrated" else "ece_best_"
        for k, v in m.items():
            row[f"{prefix}{k}"] = v

    # Save per-epoch trajectory CSV for paper learning-curve figures
    epoch_rows = extract_epoch_trajectory(log_content)
    if epoch_rows:
        epoch_path = os.path.join(log_dir, f"{name}_epochs.csv")
        _write_csv(epoch_rows, epoch_path)

    print(f"  FINISHED: {name}  ({datetime.now().isoformat()})")
    for k in ("ece", "logloss", "auc", "pcoc"):
        print(f"    loss_best_{k} = {row.get(k, 'N/A')}")
        print(f"    ece_best_{k}  = {row.get(f'ece_best_{k}', 'N/A')}")

    torch.cuda.empty_cache()

    return row


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="V7 One-Shot Experiment Runner")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index (0 or 1). Required unless --merge only.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["aliccp", "avazu"],
        help="Dataset filter for single-experiment mode.",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated experiment names to run (single-experiment / smoke-test mode).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: /root/shared-nvme/PAPER/ckpt/v7)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge gpu0.csv and gpu1.csv into summary_v7_unified.csv after both finish.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or "/root/shared-nvme/PAPER/ckpt/v7"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Merge-only mode
    # ------------------------------------------------------------------
    if args.merge and args.gpu is None:
        _merge_results(out_dir)
        return

    if args.gpu is None:
        parser.error("--gpu is required unless using --merge without --gpu.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from train_neu_ali import trial as trial_neu_ali
    from train_neu_avazu import trial as trial_neu_avazu

    def _trial(dataset, config_update):
        if dataset == "aliccp":
            return trial_neu_ali(config_update)
        else:
            return trial_neu_avazu(config_update)

    # ------------------------------------------------------------------
    # Build experiment queue
    # ------------------------------------------------------------------
    if args.methods:
        # Single-experiment / smoke-test mode
        if args.dataset is None:
            parser.error("--dataset is required when --methods is specified.")
        method_set = set(args.methods.split(","))
        queue = [
            (args.dataset, name)
            for name in method_set
            if name in ALL_EXPERIMENTS
        ]
        if not queue:
            print(f"ERROR: none of {method_set} found in ALL_EXPERIMENTS.")
            print(f"Available: {sorted(ALL_EXPERIMENTS.keys())}")
            sys.exit(1)
    else:
        queue = GPU_QUEUES[args.gpu]

    total = len(queue)
    print(f"\nV7 Experiment Runner")
    print(f"  GPU: {args.gpu}")
    print(f"  Experiments: {total}")
    print(f"  Output: {out_dir}")
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
        config_update = build_config(dataset, exp)

        # Ubins save path
        run_name = f"{dataset}_{exp_name}"
        config_update["uncertainty_bin_save_path"] = os.path.join(
            out_dir, f"{run_name}_ubins.csv"
        )

        trial_fn = (lambda cfg, ds=dataset: _trial(ds, cfg))

        row = run_single_experiment(
            trial_fn, config_update, run_name, out_dir,
            idx, total, args.gpu, dataset,
        )
        row["group"] = _get_group(exp_name)
        summary_rows.append(row)

    # ------------------------------------------------------------------
    # Write per-GPU summary CSV
    # ------------------------------------------------------------------
    suffix = f"_partial_{args.methods}" if args.methods else ""
    summary_path = os.path.join(out_dir, f"summary_v7_gpu{args.gpu}{suffix}.csv")
    _write_csv(summary_rows, summary_path)

    print(f"\n{'='*60}")
    print(f"  V7 GPU {args.gpu} COMPLETE")
    print(f"  Summary: {summary_path}")
    print(f"  Experiments: {len(summary_rows)}")
    print(f"  End: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    # Attempt merge if both CSVs exist
    if not args.methods:
        _merge_results(out_dir, silent=True)


def _get_group(exp_name):
    for e in GROUP_A:
        if e["name"] == exp_name:
            return "A"
    for e in GROUP_B:
        if e["name"] == exp_name:
            return "B"
    for e in GROUP_C:
        if e["name"] == exp_name:
            return "C"
    for e in GROUP_D:
        if e["name"] == exp_name:
            return "D"
    return "?"


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
        os.path.join(out_dir, f"summary_v7_gpu{g}.csv")
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

    unified_path = os.path.join(out_dir, "summary_v7_unified.csv")
    with open(unified_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Merged {len(all_rows)} experiments -> {unified_path}")


if __name__ == "__main__":
    main()
