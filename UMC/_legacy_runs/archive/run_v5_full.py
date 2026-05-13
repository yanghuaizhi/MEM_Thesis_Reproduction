#!/usr/bin/env python
"""V5 全量实验运行器。

并行在 2 块 GPU 上运行所有校准方法实验：
  GPU 0: AliCCP 全部方法（神经 + 统计）
  GPU 1: Avazu 全部方法（神经 + 统计）

使用方式：
  # 单 GPU 单数据集
  python run_v5_full.py --gpu 0 --dataset aliccp

  # 启动两个并行进程（推荐通过 nohup 或 screen）
  nohup python run_v5_full.py --gpu 0 --dataset aliccp > /root/shared-nvme/PAPER/ckpt/v5_full/gpu0.log 2>&1 &
  nohup python run_v5_full.py --gpu 1 --dataset avazu  > /root/shared-nvme/PAPER/ckpt/v5_full/gpu1.log 2>&1 &
"""

import os
import sys
import csv
import re
import json
import argparse
from datetime import datetime
from io import StringIO

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

# ---------------------------------------------------------------------------
# 数据集特定配置
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
# 所有方法共享的 backbone 配置
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
# 神经校准共享配置
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
    "scl_lam": 1e-3,
}

# ---------------------------------------------------------------------------
# 统计校准共享配置
# ---------------------------------------------------------------------------
STA_COMMON = {
    "batch_size_calib": 1024 * 64,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "uncertainty_bin_eval": True,
    "uncertainty_bin_num_bins": 20,
    "uncertainty_bin_ece_M": 100,
}

# ---------------------------------------------------------------------------
# 神经校准方法列表
# ---------------------------------------------------------------------------
NEU_EXPERIMENTS = [
    # 统计基线的神经等价 — 无额外参数
    {"name": "neu",        "method": "neu"},
    {"name": "desc",       "method": "desc"},
    {"name": "sbcr",       "method": "sbcr"},
    {"name": "umnn",       "method": "umnn"},
    # UMC 系列
    {"name": "umc_wor",    "method": "umc_wor"},
    {"name": "umc",        "method": "umc"},
    # UAMCM 系列
    {"name": "uamcm",      "method": "uamcm"},
    {"name": "uamcm_wor",  "method": "uamcm_wor"},
    {
        "name": "uamcm_phase4",
        "method": "uamcm_phase4",
        "ra_weighted_bce": True,
        "ra_weight_c": 1.0,
        "ra_weight_k": 1.0,
        "distill_lambda": 0.1,
        "distill_t": 1.0,
    },
    # UASAC — Stage A 最优配置 (K=3, D=16)
    {
        "name": "uasac_K3_D16",
        "method": "uasac",
        "num_experts": 3,
        "expert_dim": 16,
        "router_type": "mlp",
        "router_hidden": 64,
        "temperature": 1.0,
        "div_lambda": 0.0,
    },
]

# ---------------------------------------------------------------------------
# 统计校准方法列表
# ---------------------------------------------------------------------------
STA_EXPERIMENTS = [
    {"name": "ir",    "method": "ir"},
    {"name": "hb",    "method": "hb"},
    {"name": "platt", "method": "platt"},
]


class TeeOutput:
    """同时输出到 stdout 和 StringIO 的流封装。"""

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
    """从日志文本中提取指定 metrics_tag 后的指标行。"""
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


def run_experiments(gpu_id, dataset, out_dir):
    """运行指定数据集的全部实验。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ds_conf = DATASET_CONFIG[dataset]
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    summary_rows = []
    total_experiments = len(NEU_EXPERIMENTS) + len(STA_EXPERIMENTS)
    exp_idx = 0

    # ======================== 神经校准方法 ========================
    from train_neu_ali import trial as trial_neu_ali
    from train_neu_avazu import trial as trial_neu_avazu
    trial_neu = trial_neu_ali if dataset == "aliccp" else trial_neu_avazu

    for exp in NEU_EXPERIMENTS:
        exp_idx += 1
        name = f"{dataset}_{exp['name']}"
        method = exp["method"]

        config_update = {}
        config_update.update(BACKBONE_COMMON)
        config_update.update(ds_conf)
        config_update.update(NEU_COMMON)
        # 方法特定参数
        for k, v in exp.items():
            if k != "name":
                config_update[k] = v
        config_update["uncertainty_bin_save_path"] = os.path.join(
            out_dir, f"{name}_ubins.csv"
        )

        log_path = os.path.join(log_dir, f"{name}.log")

        print(f"\n{'='*60}")
        print(f"  [{exp_idx}/{total_experiments}] NEURAL: {name}  method={method}")
        print(f"  GPU={gpu_id}  dataset={dataset}")
        print(f"{'='*60}\n")

        old_stdout = sys.stdout
        captured = StringIO()
        sys.stdout = TeeOutput(old_stdout, captured)

        try:
            trial_neu(config_update)
        except Exception as e:
            import traceback
            print(f"EXPERIMENT_FAILED name={name} error={e}")
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout

        log_content = captured.getvalue()
        with open(log_path, "w") as f:
            f.write(log_content)

        # 提取指标
        row = {"name": name, "dataset": dataset, "type": "neural", "method": method}
        for tag in ("calibrated", "ece_best_calibrated"):
            m = extract_metrics(log_content, tag)
            if tag == "calibrated":
                for k, v in m.items():
                    row[k] = v
            else:
                for k, v in m.items():
                    row[f"ece_best_{k}"] = v
        summary_rows.append(row)

        print(f"  FINISHED: {name}")
        for k in ("ece", "logloss", "auc", "pcoc"):
            print(f"    {k} = {row.get(k, 'N/A')}")
            print(f"    ece_best_{k} = {row.get(f'ece_best_{k}', 'N/A')}")

    # ======================== 统计校准方法 ========================
    from train_sta_ali import trial as trial_sta_ali
    from train_sta_avazu import trial as trial_sta_avazu
    trial_sta = trial_sta_ali if dataset == "aliccp" else trial_sta_avazu

    for exp in STA_EXPERIMENTS:
        exp_idx += 1
        name = f"{dataset}_{exp['name']}"
        method = exp["method"]

        config_update = {}
        config_update.update(BACKBONE_COMMON)
        config_update.update(ds_conf)
        config_update.update(STA_COMMON)
        config_update["method"] = method
        config_update["uncertainty_bin_save_path"] = os.path.join(
            out_dir, f"{name}_ubins.csv"
        )

        log_path = os.path.join(log_dir, f"{name}.log")

        print(f"\n{'='*60}")
        print(f"  [{exp_idx}/{total_experiments}] STATISTICAL: {name}  method={method}")
        print(f"  GPU={gpu_id}  dataset={dataset}")
        print(f"{'='*60}\n")

        old_stdout = sys.stdout
        captured = StringIO()
        sys.stdout = TeeOutput(old_stdout, captured)

        try:
            trial_sta(config_update)
        except Exception as e:
            import traceback
            print(f"EXPERIMENT_FAILED name={name} error={e}")
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout

        log_content = captured.getvalue()
        with open(log_path, "w") as f:
            f.write(log_content)

        row = {"name": name, "dataset": dataset, "type": "statistical", "method": method}
        m = extract_metrics(log_content, "calibrated")
        for k, v in m.items():
            row[k] = v
        summary_rows.append(row)

        print(f"  FINISHED: {name}")
        for k in ("ece", "logloss", "auc", "pcoc"):
            print(f"    {k} = {row.get(k, 'N/A')}")

    # ======================== 写入 summary CSV ========================
    summary_path = os.path.join(out_dir, f"summary_{dataset}.csv")
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
    print(f"  V5 FULL EXPERIMENT COMPLETE — {dataset}")
    print(f"  Summary: {summary_path}")
    print(f"  Logs: {log_dir}")
    print(f"  Total experiments: {len(summary_rows)}")
    print(f"{'='*60}")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Full Experiment Runner")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index (0 or 1)")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aliccp", "avazu"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: /root/shared-nvme/PAPER/ckpt/v5_full_<timestamp>)",
    )
    args = parser.parse_args()

    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"/root/shared-nvme/PAPER/ckpt/v5_full_{ts}"

    os.makedirs(out_dir, exist_ok=True)
    print(f"V5 Full Experiment Runner")
    print(f"  GPU: {args.gpu}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {out_dir}")
    print(f"  Start: {datetime.now().isoformat()}")

    run_experiments(args.gpu, args.dataset, out_dir)

    print(f"  End: {datetime.now().isoformat()}")
