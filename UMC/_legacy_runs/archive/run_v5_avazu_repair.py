#!/usr/bin/env python
"""V5 Avazu OOM 修复脚本。

修复 run_v5_full.py 中 Avazu 数据集 UMC/UAMCM/UASAC 系列方法因
CUDA OOM 导致的失败。

根本原因：
  ParallelNeuralIntegral 在计算 UMC/UAMCM/UASAC 前向传播时，需要将
  batch 展开为 batch * (nb_steps+1) 个并发评估点。Avazu backbone
  (PackedDeepFM, 16 estimators, 21 high-cardinality fields) 占用约
  21.5 GiB 显存，剩余 ~2.1 GiB 不足以容纳 batch=65536 时
  ParallelNeuralIntegral 所需的 ~4.18 GiB 激活张量。

修复方案：
  将 batch_size_calib 从 65536 降至 16384（4× 缩减），
  h_steps 等中间张量降至 ~1 GiB，可以安全地在剩余显存中运行。

使用方式：
  # 在 Avazu 所在 GPU（GPU 1）上运行修复
  python run_v5_avazu_repair.py --gpu 1

  # 后台运行（推荐）
  nohup python run_v5_avazu_repair.py --gpu 1 \
    > /root/shared-nvme/PAPER/ckpt/v5_full/gpu1_avazu_repair.log 2>&1 &
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
# 与 run_v5_full.py 保持一致的基础配置
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

DATASET_CONFIG = {
    "data_name": "avazu",
    "field_index": 2,
}

# 修复关键：将 batch_size_calib 从 65536 降到 16384
# ParallelNeuralIntegral 内存消耗 ∝ batch_size × nb_steps
# 65536 → OOM (~4.18 GiB)；16384 → ~1.05 GiB，可安全运行
NEU_COMMON_REPAIR = {
    "lr_calib": 1e-3,
    "epochs_calib": 20,
    "batch_size_calib": 1024 * 16,   # <<< 修复点：原为 1024*64=65536
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

# 只重跑 OOM 失败的 6 个实验
REPAIR_EXPERIMENTS = [
    {"name": "umc_wor",    "method": "umc_wor"},
    {"name": "umc",        "method": "umc"},
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

OUT_DIR = "/root/shared-nvme/PAPER/ckpt/v5_full"


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
        if f"metrics_tag={tag}" in line:
            found_tag = True
            continue
        if found_tag and "test_" in line:
            for m in re.finditer(r"test_(\w+)\s*=\s*([\d.eE+-]+)", line):
                metrics[m.group(1)] = m.group(2)
            break
    return metrics


def update_summary_csv(summary_path, new_rows_by_name):
    """读取现有 CSV，更新指定行，写回。"""
    if not os.path.exists(summary_path):
        print(f"  [WARN] summary CSV 不存在：{summary_path}，跳过更新")
        return

    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # 更新对应行
    updated = set()
    for row in rows:
        name = row.get("name", "")
        if name in new_rows_by_name:
            new_data = new_rows_by_name[name]
            for k, v in new_data.items():
                if k in row or k not in fieldnames:
                    row[k] = v
                    if k not in fieldnames:
                        fieldnames.append(k)
            updated.add(name)

    # 将未出现在原有 CSV 中的行追加
    for name, data in new_rows_by_name.items():
        if name not in updated:
            rows.append(data)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  summary CSV 已更新：{summary_path}")


def run_repair(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir = os.path.join(OUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    from train_neu_avazu import trial as trial_neu

    dataset = "avazu"
    total = len(REPAIR_EXPERIMENTS)
    new_rows = {}  # name → row dict（用于更新 summary CSV）

    print(f"\nV5 Avazu OOM 修复运行器")
    print(f"  GPU: {gpu_id}")
    print(f"  batch_size_calib: {NEU_COMMON_REPAIR['batch_size_calib']} (原 65536)")
    print(f"  修复实验数: {total}")
    print(f"  Start: {datetime.now().isoformat()}")

    for idx, exp in enumerate(REPAIR_EXPERIMENTS, 1):
        exp_name = exp["name"]
        name = f"{dataset}_{exp_name}"
        method = exp["method"]

        config_update = {}
        config_update.update(BACKBONE_COMMON)
        config_update.update(DATASET_CONFIG)
        config_update.update(NEU_COMMON_REPAIR)
        for k, v in exp.items():
            if k != "name":
                config_update[k] = v
        config_update["uncertainty_bin_save_path"] = os.path.join(
            OUT_DIR, f"{name}_ubins.csv"
        )
        # 记录修复标记（日志可查）
        config_update["_repair_batch_size_calib_original"] = 65536

        log_path = os.path.join(log_dir, f"{name}.log")

        print(f"\n{'='*60}")
        print(f"  [REPAIR {idx}/{total}] NEURAL: {name}  method={method}")
        print(f"  GPU={gpu_id}  dataset={dataset}")
        print(f"  batch_size_calib={NEU_COMMON_REPAIR['batch_size_calib']}")
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

        # 提取指标，构建 summary 行
        row = {"name": name, "dataset": dataset, "type": "neural", "method": method}
        for tag in ("calibrated", "ece_best_calibrated"):
            m = extract_metrics(log_content, tag)
            if tag == "calibrated":
                for k, v in m.items():
                    row[k] = v
            else:
                for k, v in m.items():
                    row[f"ece_best_{k}"] = v
        new_rows[name] = row

        print(f"  FINISHED: {name}")
        for k in ("ece", "logloss", "auc", "pcoc"):
            print(f"    {k} = {row.get(k, 'N/A')}")
            print(f"    ece_best_{k} = {row.get(f'ece_best_{k}', 'N/A')}")

        # 每轮结束后释放显存碎片
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 更新 summary CSV
    summary_path = os.path.join(OUT_DIR, "summary_avazu.csv")
    update_summary_csv(summary_path, new_rows)

    print(f"\n{'='*60}")
    print(f"  REPAIR COMPLETE — avazu ({total} experiments)")
    print(f"  Summary updated: {summary_path}")
    print(f"  Logs: {log_dir}")
    print(f"  End: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Avazu OOM 修复运行器")
    parser.add_argument(
        "--gpu", type=int, default=1,
        help="GPU index（Avazu 默认使用 GPU 1）"
    )
    args = parser.parse_args()
    run_repair(args.gpu)
