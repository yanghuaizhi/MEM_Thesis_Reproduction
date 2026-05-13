#!/usr/bin/env python
"""Re-run 5 missing V7 supplement experiments and rebuild summary CSVs.

Missing experiments:
  GPU0: aliccp_desc_s1024         (~84 min)
  GPU1: avazu_uasac_r_K3_s1024    (~28 min)
        avazu_ir / avazu_hb / avazu_platt  (~5 min each)

Usage:
  # Run re-runs on two GPUs (in parallel via nohup):
  CUDA_VISIBLE_DEVICES=0 python rerun_missing.py --gpu 0
  CUDA_VISIBLE_DEVICES=1 python rerun_missing.py --gpu 1

  # After both complete — rebuild summary CSVs from ALL logs:
  python rerun_missing.py --rebuild
"""

import os
import sys
import argparse
from datetime import datetime

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

# Reuse all infrastructure from run_v7_supplement
from run_v7_supplement import (
    ALL_EXPERIMENTS,
    GPU_QUEUES,
    build_config_neu,
    build_config_sta,
    run_single_experiment,
    extract_metrics,
    _write_csv,
    _merge_results,
    _merge_v7,
    DATASET_CONFIG,
)

from train_neu_ali import trial as trial_neu_ali
from train_sta_ali import trial as trial_sta_ali
from train_neu_avazu import trial as trial_neu_avazu
from train_sta_avazu import trial as trial_sta_avazu

TRIAL_FNS = {
    ("aliccp", "neu"): trial_neu_ali,
    ("aliccp", "sta"): trial_sta_ali,
    ("avazu", "neu"): trial_neu_avazu,
    ("avazu", "sta"): trial_sta_avazu,
}

OUT_DIR = "/root/shared-nvme/PAPER/ckpt/v7_supp"

# Missing experiments per GPU
RERUN_QUEUES = {
    0: [
        ("aliccp", "desc_s1024"),
    ],
    1: [
        ("avazu", "uasac_r_K3_s1024"),
        ("avazu", "ir"),
        ("avazu", "hb"),
        ("avazu", "platt"),
    ],
}


def run_rerun(gpu_id):
    """Run the missing experiments for the specified GPU."""
    import torch
    torch.cuda.set_device(0)  # local device 0 (CUDA_VISIBLE_DEVICES selects physical GPU)

    queue = RERUN_QUEUES[gpu_id]
    total = len(queue)

    print(f"\n{'='*60}")
    print(f"  V7 Supplement RERUN — GPU {gpu_id}")
    print(f"  Experiments: {total}")
    print(f"  Queue: {queue}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    for idx, (dataset, exp_name) in enumerate(queue, 1):
        exp = ALL_EXPERIMENTS[exp_name]
        trial_type = exp["trial_type"]

        if trial_type == "neu":
            config_update = build_config_neu(dataset, exp)
        else:
            config_update = build_config_sta(dataset, exp)

        run_name = f"{dataset}_{exp_name}"
        config_update["uncertainty_bin_save_path"] = os.path.join(
            OUT_DIR, f"{run_name}_ubins.csv"
        )

        trial_fn = TRIAL_FNS[(dataset, trial_type)]

        row = run_single_experiment(
            trial_fn, config_update, run_name, OUT_DIR,
            idx, total, gpu_id, dataset, trial_type,
        )

        # Verify metrics were extracted
        has_metrics = bool(row.get("auc"))
        print(f"  Metrics extracted: {has_metrics}")

    print(f"\n{'='*60}")
    print(f"  V7 Supplement RERUN GPU {gpu_id} COMPLETE")
    print(f"  End: {datetime.now().isoformat()}")
    print(f"{'='*60}")


def rebuild_summaries():
    """Rebuild GPU0/GPU1 summary CSVs from ALL per-experiment logs."""
    import re
    import csv
    import glob

    log_dir = os.path.join(OUT_DIR, "logs")

    # Map each experiment name to its original GPU queue
    exp_to_gpu = {}
    for gpu_id, queue in GPU_QUEUES.items():
        for dataset, exp_name in queue:
            run_name = f"{dataset}_{exp_name}"
            exp_to_gpu[run_name] = gpu_id

    # Parse all logs
    gpu_rows = {0: [], 1: []}
    all_logs = sorted(glob.glob(os.path.join(log_dir, "*.log")))

    print(f"\nRebuilding summaries from {len(all_logs)} log files...")

    for logf in all_logs:
        name = os.path.basename(logf).replace(".log", "")

        with open(logf) as f:
            log_content = f.read()

        # Parse dataset and experiment name
        dataset = None
        for ds in ["aliccp", "avazu"]:
            if name.startswith(ds + "_"):
                dataset = ds
                break
        if not dataset:
            print(f"  SKIP (unknown dataset): {name}")
            continue

        exp_name = name[len(dataset) + 1:]

        # Determine trial type
        trial_type = "sta" if exp_name in ["ir", "hb", "platt"] else "neu"

        # Determine method from ALL_EXPERIMENTS if possible
        if exp_name in ALL_EXPERIMENTS:
            exp_def = ALL_EXPERIMENTS[exp_name]
            method = exp_def["method"]
        else:
            method = exp_name

        # Extract metrics
        cal_metrics = extract_metrics(log_content, "calibrated")
        ece_metrics = extract_metrics(log_content, "ece_best_calibrated")

        if not cal_metrics:
            print(f"  WARN: No calibrated metrics in {name}")
            continue

        # Build row
        row = {
            "name": name,
            "dataset": dataset,
            "method": method,
            "trial_type": trial_type,
        }

        # Add calib_seed if neural
        if trial_type == "neu":
            seed_match = re.search(r"_s(\d+)$", exp_name)
            row["calib_seed"] = int(seed_match.group(1)) if seed_match else "N/A"

            # Get router_mode and div_lambda from experiment definition
            if exp_name in ALL_EXPERIMENTS:
                exp_def = ALL_EXPERIMENTS[exp_name]
                config = build_config_neu(dataset, exp_def)
                row["router_mode"] = config.get("router_mode", "N/A")
                row["div_lambda"] = config.get("div_lambda", "N/A")
                row["scl_lam"] = config.get("scl_lam", "N/A")
            else:
                row["router_mode"] = "N/A"
                row["div_lambda"] = "N/A"
                row["scl_lam"] = "N/A"
        else:
            row["calib_seed"] = "N/A"

        # Add calibrated metrics
        for k, v in cal_metrics.items():
            row[k] = v

        # Add ece_best metrics
        for k, v in ece_metrics.items():
            row[f"ece_best_{k}"] = v

        # Assign to GPU
        gpu_id = exp_to_gpu.get(name, None)
        if gpu_id is not None:
            gpu_rows[gpu_id].append(row)
        else:
            print(f"  WARN: {name} not in any GPU queue, adding to GPU1")
            gpu_rows[1].append(row)

    # Write GPU summaries
    for gpu_id in [0, 1]:
        rows = gpu_rows[gpu_id]
        path = os.path.join(OUT_DIR, f"summary_v7_supp_gpu{gpu_id}.csv")
        if rows:
            _write_csv(rows, path)
            print(f"  GPU{gpu_id}: {len(rows)} experiments -> {path}")
        else:
            print(f"  GPU{gpu_id}: No rows!")

    # Merge
    print("\nRunning merge (unified)...")
    _merge_results(OUT_DIR)

    print("\nRunning merge-v7 (mean±std)...")
    _merge_v7(OUT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Re-run missing V7 supplement experiments")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (0 or 1)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild summary CSVs from all logs and run merge")
    args = parser.parse_args()

    if args.rebuild:
        rebuild_summaries()
    elif args.gpu is not None:
        run_rerun(args.gpu)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
