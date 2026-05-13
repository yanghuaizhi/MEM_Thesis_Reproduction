"""Stage A: Lightweight validation of UASAC on AliCCP.

Runs 5 experiments sequentially:
  S-A1: umc_wor baseline (sanity check)
  S-A2: uasac K=3 D_exp=16 default
  S-A3: uasac K=3 D_exp=16 div_lambda=0.01
  S-A4: uasac K=5 D_exp=16
  S-A5: uasac K=3 D_exp=32
"""

import os
import sys
import csv
import re
from datetime import datetime
from io import StringIO

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

from train_neu_ali import trial

COMMON = {
    "data_root": "/root/shared-nvme/PAPER/dataset",
    "filepath": "/root/shared-nvme/PAPER/ckpt",
    "data_name": "aliccp",
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
    "field_index": 0,
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
}

EXPERIMENTS = [
    {
        "name": "SA1_umc_wor",
        "method": "umc_wor",
        "scl_lam": 1e-3,
    },
    {
        "name": "SA2_uasac_K3_D16",
        "method": "uasac",
        "num_experts": 3,
        "expert_dim": 16,
        "router_type": "mlp",
        "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.0,
        "scl_lam": 1e-3,
    },
    {
        "name": "SA3_uasac_K3_D16_div",
        "method": "uasac",
        "num_experts": 3,
        "expert_dim": 16,
        "router_type": "mlp",
        "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.01,
        "scl_lam": 1e-3,
    },
    {
        "name": "SA4_uasac_K5_D16",
        "method": "uasac",
        "num_experts": 5,
        "expert_dim": 16,
        "router_type": "mlp",
        "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.0,
        "scl_lam": 1e-3,
    },
    {
        "name": "SA5_uasac_K3_D32",
        "method": "uasac",
        "num_experts": 3,
        "expert_dim": 32,
        "router_type": "mlp",
        "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.0,
        "scl_lam": 1e-3,
    },
]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/root/shared-nvme/PAPER/ckpt/stage_a_{ts}"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

summary_rows = []

for exp in EXPERIMENTS:
    name = exp.pop("name")
    config_update = {**COMMON}
    config_update.update(exp)
    config_update["uncertainty_bin_save_path"] = os.path.join(out_dir, f"{name}_ubins.csv")

    log_path = os.path.join(out_dir, "logs", f"{name}.log")

    print(f"\n{'='*60}")
    print(f"  STARTING: {name}  method={config_update['method']}")
    print(f"{'='*60}\n")

    old_stdout = sys.stdout
    captured = StringIO()

    class TeeOutput:
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

    sys.stdout = TeeOutput(old_stdout, captured)

    try:
        trial(config_update)
    except Exception as e:
        print(f"EXPERIMENT_FAILED name={name} error={e}")
    finally:
        sys.stdout = old_stdout

    log_content = captured.getvalue()
    with open(log_path, "w") as f:
        f.write(log_content)

    row = {"name": name, "method": config_update["method"]}
    for key in ("num_experts", "expert_dim", "div_lambda"):
        row[key] = config_update.get(key, "")

    calib_lines = []
    for line in log_content.split("\n"):
        if "metrics_tag=calibrated" in line:
            calib_lines.append("calibrated")
        elif calib_lines and calib_lines[-1] == "calibrated" and "test_ece" in line:
            calib_lines[-1] = line
    if calib_lines:
        last = calib_lines[-1]
        for m in re.finditer(r"test_(\w+)\s*=\s*([\d.eE+-]+)", last):
            row[m.group(1)] = m.group(2)

    summary_rows.append(row)
    print(f"\n  FINISHED: {name}")
    for k in ("ece", "logloss", "auc", "pcoc"):
        print(f"    {k} = {row.get(k, 'N/A')}")

summary_path = os.path.join(out_dir, "summary.csv")
if summary_rows:
    all_keys = list(summary_rows[0].keys())
    for r in summary_rows[1:]:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(summary_rows)

print(f"\n{'='*60}")
print(f"  STAGE A COMPLETE — Summary: {summary_path}")
print(f"{'='*60}")
