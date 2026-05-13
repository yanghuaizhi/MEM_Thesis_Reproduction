#!/usr/bin/env python
"""V8 Validation runner — Minimal single-seed validation for two new innovation points.

New methods (all seed=1024):
  uamcm_no_u_rs      : UAMCM-D (u only in integrand, feature-only rescaling)
  uamcm_dascl        : UAMCM + DA-SCL (density-aligned u-stratified calib loss)
  uamcm_no_u_rs_dascl: UAMCM-D + DA-SCL (combined)

Usage:
  # Run all 9 experiments (3 methods × 3 datasets, single GPU, ~6h):
  python run_v8_validation.py --gpu 0

  # Run for a single dataset only:
  python run_v8_validation.py --gpu 0 --dataset aliccp
  python run_v8_validation.py --gpu 0 --dataset avazu
  python run_v8_validation.py --gpu 0 --dataset criteo

  # Smoke test (only uamcm_no_u_rs on aliccp):
  python run_v8_validation.py --gpu 0 --smoke

Output: ckpt/v8_validation/
"""

import os
import sys
import csv
import argparse
import torch
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

# ---------------------------------------------------------------------------
# Shared backbone config (identical across all datasets)
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
# Dataset-specific config
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "aliccp": {"data_name": "aliccp", "field_index": 0},
    "avazu":  {"data_name": "avazu",  "field_index": 2},
    "criteo": {"data_name": "criteo", "field_index": 23},
}

# OOM-safe batch sizes for memory-intensive methods
SMALL_BATCH_METHODS = {"uamcm_no_u_rs", "uamcm_dascl", "uamcm_no_u_rs_dascl"}
BATCH_SIZES = {
    "aliccp": 1024 * 64,   # standard
    "avazu":  1024 * 16,   # avazu needs smaller batch
    "criteo": 1024 * 32,   # criteo needs smaller batch
}

# ---------------------------------------------------------------------------
# New experiment definitions (single seed=1024, all 3 methods)
# ---------------------------------------------------------------------------
_DASCL_LAM = 0.01
_DASCL_BINS = 10

NEW_METHODS = [
    {
        "name": "uamcm_no_u_rs",
        "method": "uamcm_no_u_rs",
        "calib_seed": 1024,
    },
    {
        "name": "uamcm_dascl",
        "method": "uamcm_dascl",
        "calib_seed": 1024,
        "dascl_lam": _DASCL_LAM,
        "dascl_bins": _DASCL_BINS,
    },
    {
        "name": "uamcm_no_u_rs_dascl",
        "method": "uamcm_no_u_rs_dascl",
        "calib_seed": 1024,
        "dascl_lam": _DASCL_LAM,
        "dascl_bins": _DASCL_BINS,
    },
]

ALL_DATASETS = ["aliccp", "avazu", "criteo"]


def build_config(dataset, exp):
    cfg = {}
    cfg.update(BACKBONE_COMMON)
    cfg.update(NEU_COMMON)
    cfg.update(DATASET_CONFIG[dataset])
    cfg.update(exp)
    # Adjust batch size
    if exp["method"] in SMALL_BATCH_METHODS:
        cfg["batch_size_calib"] = BATCH_SIZES[dataset]
    return cfg


# ---------------------------------------------------------------------------
# TeeOutput utility (copy from run_v7.py)
# ---------------------------------------------------------------------------
class TeeOutput:
    def __init__(self, orig, log_file):
        self.orig = orig
        self.log = log_file

    def write(self, data):
        self.orig.write(data)
        self.orig.flush()
        try:
            self.log.write(data)
            self.log.flush()
            os.fsync(self.log.fileno())
        except Exception:
            pass

    def flush(self):
        self.orig.flush()

    def fileno(self):
        return self.orig.fileno()


def run_single_experiment(trial_fn, config_update, run_name, out_dir, idx, total, gpu, dataset):
    log_path = os.path.join(out_dir, "logs", f"{run_name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{idx}/{total}] {run_name}")
    print(f"  method={config_update.get('method')} dataset={dataset}")
    print(f"  dascl_lam={config_update.get('dascl_lam', 0.0)} seed={config_update.get('calib_seed')}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    import sys as _sys
    orig_stdout = _sys.stdout
    orig_stderr = _sys.stderr
    with open(log_path, "w") as log_file:
        tee_out = TeeOutput(orig_stdout, log_file)
        tee_err = TeeOutput(orig_stderr, log_file)
        _sys.stdout = tee_out
        _sys.stderr = tee_err
        try:
            result = trial_fn(config_update)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            result = {}
        finally:
            _sys.stdout = orig_stdout
            _sys.stderr = orig_stderr

    row = {"name": run_name, "dataset": dataset, "gpu": gpu}
    row.update(config_update)
    if isinstance(result, dict):
        row.update(result)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["aliccp", "avazu", "criteo", None],
                        help="Run only specified dataset. Default: all 3.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run only uamcm_no_u_rs on aliccp.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)

    out_dir = "/root/shared-nvme/PAPER/ckpt/v8_validation"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    # Import trial functions
    from train_neu_ali import trial as trial_ali
    from train_neu_avazu import trial as trial_avazu
    from train_neu_criteo import trial as trial_criteo

    trial_fns = {
        "aliccp": trial_ali,
        "avazu": trial_avazu,
        "criteo": trial_criteo,
    }

    # Build experiment queue
    if args.smoke:
        queue = [("aliccp", NEW_METHODS[0])]  # only uamcm_no_u_rs on aliccp
    else:
        datasets = [args.dataset] if args.dataset else ALL_DATASETS
        queue = [(ds, exp) for ds in datasets for exp in NEW_METHODS]

    total = len(queue)
    print(f"\nV8 Validation Runner")
    print(f"  GPU: {args.gpu}")
    print(f"  Experiments: {total}")
    print(f"  Output: {out_dir}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"  Methods: {[e['name'] for e in NEW_METHODS]}")

    summary_rows = []
    for idx, (dataset, exp) in enumerate(queue, 1):
        config_update = build_config(dataset, exp)
        run_name = f"{dataset}_{exp['name']}"
        config_update["uncertainty_bin_save_path"] = os.path.join(
            out_dir, f"{run_name}_ubins.csv"
        )

        trial_fn = trial_fns[dataset]
        row = run_single_experiment(
            trial_fn, config_update, run_name, out_dir,
            idx, total, args.gpu, dataset,
        )
        summary_rows.append(row)

    summary_path = os.path.join(out_dir, "summary_v8_validation.csv")
    _write_csv(summary_rows, summary_path)

    print(f"\n{'='*60}")
    print(f"  V8 Validation COMPLETE")
    print(f"  Summary: {summary_path}")
    print(f"  Experiments: {len(summary_rows)}")
    print(f"  End: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
