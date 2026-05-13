#!/usr/bin/env python3
"""
run_v10_ablation2.py -- Ablation 2: u signal source analysis
=============================================================

Purpose:
  Test whether UAMCM's improvement comes from PE variance's information content
  or merely from the extra input dimension / base-rate correlation.

  Four conditions (UMC and UAMCM(u=PE) already exist in V7/V9 results):
    - UMC:                 no u input (baseline, from existing results)
    - UAMCM(u=PE):         original PE variance (from existing results)
    - UAMCM(u=shuffled):   permuted u (destroys sample-level correlation) [NEW]
    - UAMCM(u=logit_in):   normalized logit_in as u proxy [NEW]

Output:
  ckpt/v10_ablation2/
    ubins/     -- per-uncertainty-bin aggregated CSV
    logs/      -- full training logs (stdout + stderr)
    epochs/    -- per-epoch training curves CSV
    configs/   -- full config JSON (reproducible)
    summary/   -- summary CSV

Usage:
  python run_v10_ablation2.py                     # 18 experiments (default)
  python run_v10_ablation2.py --smoke             # 2 experiments, 1 epoch
  python run_v10_ablation2.py --criteo-fallback   # Criteo reduced batch + baselines
  python run_v10_ablation2.py --dataset avazu     # filter by dataset
  python run_v10_ablation2.py --dry-run           # preview only
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from time import time

# -- Path config -------------------------------------------------------

PAPER_ROOT = "/root/shared-nvme/PAPER"
CKPT_ROOT = os.path.join(PAPER_ROOT, "ckpt")
OUT_DIR = os.path.join(CKPT_ROOT, "v10_ablation2")
UBINS_DIR = os.path.join(OUT_DIR, "ubins")
LOG_DIR = os.path.join(OUT_DIR, "logs")
EPOCH_DIR = os.path.join(OUT_DIR, "epochs")
CONFIG_DIR = os.path.join(OUT_DIR, "configs")
SUMMARY_DIR = os.path.join(OUT_DIR, "summary")

for d in [OUT_DIR, UBINS_DIR, LOG_DIR, EPOCH_DIR, CONFIG_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)

# -- Import trial functions --------------------------------------------

sys.path.insert(0, os.path.join(PAPER_ROOT, "UMC"))

from train_neu_ali import trial as trial_ali
from train_neu_avazu import trial as trial_avazu

try:
    from train_neu_criteo import trial as trial_criteo
    HAS_CRITEO = True
except ImportError:
    HAS_CRITEO = False

# -- Backbone config (identical to V7/V9) -----------------------------

BACKBONE_COMMON = {
    "model_name": "packed_deepfm",
    "batch_size": 32768,
    "dropout": 0.1,
    "init_std": 0.0001,
    "lr": 0.0005,
    "l2_reg": 1e-5,
    "seed": 1024,
    "num_estimators": 16,
    "alpha": 1.0,
    "gamma": 1,
    "data_root": os.path.join(PAPER_ROOT, "dataset"),
    "filepath": CKPT_ROOT,
}

CALIB_COMMON = {
    "lr_calib": 1e-3,
    "epochs_calib": 20,
    "scl_lam": 0.01,
    "scl_beta": 0.95,
    "calib_early_stop": True,
    "calib_patience": 5,
    "calib_min_delta": 1e-5,
    "calib_restore_best": True,
    "calib_log_every": 200,
    "num_workers": 4,
    "persistent_workers": True,
    "pin_memory": True,
    "ece_M": 100,
    "uncertainty_bin_eval": True,
    "uncertainty_bin_num_bins": 20,
    "uncertainty_bin_ece_M": 100,
}

DATASET_CONFIG = {
    "aliccp": {"data_name": "aliccp", "batch_size_calib": 1024 * 64, "field_index": 0},
    "avazu":  {"data_name": "avazu",  "batch_size_calib": 1024 * 16, "field_index": 2},
    "criteo": {"data_name": "criteo", "batch_size_calib": 1024 * 32, "field_index": 0},
}

CRITEO_FALLBACK_BATCH = 1024 * 16

# -- Experiment definitions --------------------------------------------

def make_experiments(dataset_filter=None, seed_filter=None,
                     u_mode_filter=None, criteo_fallback=False, smoke=False):
    if smoke:
        return [
            {
                "name": "smoke_avazu_uamcm_shuffled_s1024",
                "dataset": "avazu",
                "method": "uamcm",
                "u_mode": "shuffled",
                "calib_seed": 1024,
            },
            {
                "name": "smoke_avazu_uamcm_logit_s1024",
                "dataset": "avazu",
                "method": "uamcm",
                "u_mode": "logit",
                "calib_seed": 1024,
            },
        ]

    experiments = []
    datasets = ["aliccp", "avazu", "criteo"] if HAS_CRITEO else ["aliccp", "avazu"]
    seeds = [1024, 2024, 3024]

    if dataset_filter:
        datasets = [d for d in datasets if d == dataset_filter]
    if seed_filter:
        seeds = [s for s in seeds if s == seed_filter]
    if u_mode_filter:
        u_modes = [u_mode_filter]
    else:
        u_modes = None

    if criteo_fallback:
        # Criteo only: all 4 conditions with reduced batch size
        conditions = [
            ("umc", "pe"),
            ("uamcm", "pe"),
            ("uamcm", "shuffled"),
            ("uamcm", "logit"),
        ]
        for method, u_mode in conditions:
            for seed in seeds:
                experiments.append({
                    "name": f"abl2_criteo_{method}_{u_mode}_s{seed}",
                    "dataset": "criteo",
                    "method": method,
                    "u_mode": u_mode,
                    "calib_seed": seed,
                })
    else:
        # Standard: 2 new u_modes x all datasets x all seeds
        for u_mode in (u_modes or ["shuffled", "logit"]):
            for ds in datasets:
                for seed in seeds:
                    experiments.append({
                        "name": f"abl2_{ds}_uamcm_{u_mode}_s{seed}",
                        "dataset": ds,
                        "method": "uamcm",
                        "u_mode": u_mode,
                        "calib_seed": seed,
                    })

    return experiments


def build_config(exp, smoke=False, run_id="", criteo_fallback=False):
    config = {}
    config.update(BACKBONE_COMMON)
    config.update(CALIB_COMMON)
    config.update(DATASET_CONFIG[exp["dataset"]])
    config["method"] = exp["method"]
    config["calib_seed"] = exp["calib_seed"]
    config["u_mode"] = exp["u_mode"]

    # UMC does not consume u — u_mode only meaningful for UAMCM variants
    if exp["method"] == "umc" and exp["u_mode"] != "pe":
        raise ValueError(
            f"method=umc does not use u, u_mode={exp['u_mode']!r} is meaningless. "
            f"Use u_mode='pe' for UMC baseline."
        )

    if criteo_fallback and exp["dataset"] == "criteo":
        config["batch_size_calib"] = CRITEO_FALLBACK_BATCH

    if smoke:
        config["epochs_calib"] = 1
        config["calib_early_stop"] = False

    run_name = f"{exp['dataset']}_{exp['method']}_{exp['u_mode']}_s{exp['calib_seed']}"
    file_tag = f"{run_name}_{run_id}" if run_id else run_name
    config["uncertainty_bin_save_path"] = os.path.join(UBINS_DIR, f"{file_tag}_ubins.csv")

    return config, run_name, file_tag


# -- Epoch trajectory extraction (reused from V9) ---------------------

def extract_epoch_trajectory(log_content):
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
            for j in range(i + 1, min(i + 5, len(lines))):
                m_ece = re.search(
                    r"ece_track epoch=\d+ test_ece=([\d.eE+-]+)", lines[j]
                )
                if m_ece:
                    test_metrics["ece"] = float(m_ece.group(1))
                    break
            for j in range(i + 1, min(i + 10, len(lines))):
                if "metrics_tag=calibrated_fast" in lines[j]:
                    for k in range(j + 1, min(j + 5, len(lines))):
                        for mm in re.finditer(
                            r"test_(\w+)\s*=\s*([\d.eE+-]+)", lines[k]
                        ):
                            test_metrics[mm.group(1)] = float(mm.group(2))
                        if test_metrics.get("auc"):
                            break
                    break
            rows.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "best_train_loss": best_loss,
                "best_epoch": best_epoch,
                "test_ece": test_metrics.get("ece", ""),
                "test_logloss": test_metrics.get("logloss", ""),
                "test_auc": test_metrics.get("auc", ""),
                "test_pcoc": test_metrics.get("pcoc", ""),
            })
        i += 1
    return rows


# -- TeeStream --------------------------------------------------------

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# -- Main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V10 Ablation 2: u signal source")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter by dataset (aliccp/avazu/criteo)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Filter by calib seed")
    parser.add_argument("--u-mode", type=str, default=None,
                        choices=["shuffled", "logit"],
                        help="Filter by u_mode (shuffled/logit)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test (2 experiments, 1 epoch)")
    parser.add_argument("--criteo-fallback", action="store_true",
                        help="Criteo reduced batch + run all 4 conditions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview experiments without running")
    args = parser.parse_args()

    experiments = make_experiments(
        dataset_filter=args.dataset,
        seed_filter=args.seed,
        u_mode_filter=args.u_mode,
        criteo_fallback=args.criteo_fallback,
        smoke=args.smoke,
    )

    if not experiments:
        print("No experiments match the filter criteria.")
        return

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    print("V10 Ablation 2: u Signal Source Analysis")
    print(f"  Run ID: {run_id}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Smoke: {args.smoke}")
    print(f"  Criteo fallback: {args.criteo_fallback}")
    if args.criteo_fallback:
        print(f"  Criteo batch_size_calib: {CRITEO_FALLBACK_BATCH}")
    print(f"  Start: {datetime.now().isoformat()}")
    print()

    for i, exp in enumerate(experiments):
        batch_note = ""
        if args.criteo_fallback and exp["dataset"] == "criteo":
            batch_note = f"  batch={CRITEO_FALLBACK_BATCH}"
        print(f"  [{i+1}/{len(experiments)}] {exp['name']}  "
              f"dataset={exp['dataset']}  method={exp['method']}  "
              f"u_mode={exp['u_mode']}  seed={exp['calib_seed']}{batch_note}")

    if args.dry_run:
        print("\n[DRY RUN] Would run above experiments. Exiting.")
        return

    print()

    # Run experiments
    results = []
    for i, exp in enumerate(experiments):
        config, run_name, file_tag = build_config(
            exp, smoke=args.smoke, run_id=run_id,
            criteo_fallback=args.criteo_fallback,
        )

        # Save config JSON
        config_path = os.path.join(CONFIG_DIR, f"{file_tag}.json")
        config_record = {
            "run_name": run_name,
            "file_tag": file_tag,
            "run_id": run_id,
            "experiment": exp,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "smoke": args.smoke,
            "criteo_fallback": args.criteo_fallback,
        }
        with open(config_path, "w") as f:
            json.dump(config_record, f, indent=2)

        # Tee stdout/stderr to log file
        log_path = os.path.join(LOG_DIR, f"{file_tag}.log")
        log_file = open(log_path, "w")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeStream(old_stdout, log_file)
        sys.stderr = TeeStream(old_stderr, log_file)

        print("=" * 60)
        print(f"  [{i+1}/{len(experiments)}] {exp['name']}")
        print(f"  dataset={exp['dataset']}  method={exp['method']}  "
              f"u_mode={exp['u_mode']}  seed={exp['calib_seed']}")
        print(f"  batch_size_calib={config['batch_size_calib']}")
        print(f"  Start: {datetime.now().isoformat()}")
        print(f"  Config: {config_path}")
        print("=" * 60)

        t0 = time()
        try:
            trial_fn = {
                "aliccp": trial_ali,
                "avazu": trial_avazu,
                "criteo": trial_criteo if HAS_CRITEO else None,
            }[exp["dataset"]]

            if trial_fn is None:
                print(f"  SKIP: {exp['dataset']} trial function not available")
                continue

            metrics = trial_fn(config)
            elapsed = time() - t0
            print(f"\n  FINISHED: {run_name}  ({elapsed:.0f}s)")
            print(f"  Metrics: {json.dumps(metrics, indent=2, default=str)}")

            results.append({
                "name": run_name,
                **exp,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "elapsed_s": elapsed,
                "status": "ok",
            })
        except Exception as e:
            elapsed = time() - t0
            print(f"\n  ERROR: {run_name}  ({elapsed:.0f}s)")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": run_name,
                **exp,
                "elapsed_s": elapsed,
                "status": f"error: {e}",
            })

        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()

        # Extract per-epoch trajectory
        with open(log_path, "r") as f:
            log_content = f.read()
        epoch_rows = extract_epoch_trajectory(log_content)
        if epoch_rows:
            import csv
            epoch_path = os.path.join(EPOCH_DIR, f"{file_tag}_epochs.csv")
            with open(epoch_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=epoch_rows[0].keys())
                writer.writeheader()
                writer.writerows(epoch_rows)
            print(f"  Epoch trajectory: {epoch_path} ({len(epoch_rows)} epochs)")

        # List output artifacts
        print(f"  Artifacts:")
        for label, path in [
            ("config", config_path),
            ("log", log_path),
            ("epochs", os.path.join(EPOCH_DIR, f"{file_tag}_epochs.csv")),
            ("ubins", os.path.join(UBINS_DIR, f"{file_tag}_ubins.csv")),
            ("ubins_ece", os.path.join(UBINS_DIR, f"{file_tag}_ubins_ece_best.csv")),
        ]:
            exists = os.path.exists(path)
            size = ""
            if exists:
                size_bytes = os.path.getsize(path)
                if size_bytes > 1024 * 1024:
                    size = f" ({size_bytes / 1024 / 1024:.1f}MB)"
                elif size_bytes > 1024:
                    size = f" ({size_bytes / 1024:.0f}KB)"
            status = "OK" if exists else "MISSING"
            print(f"    {label:12s}: {status}{size}")
        print()

    # Save summary
    import pandas as pd
    df = pd.DataFrame(results)
    tag = "fallback" if args.criteo_fallback else "default"
    summary_path = os.path.join(SUMMARY_DIR, f"summary_v10_abl2_{tag}_{run_id}.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    print(df.to_string(index=False))
    total_time = sum(r.get("elapsed_s", 0) for r in results)
    print(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"End: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
