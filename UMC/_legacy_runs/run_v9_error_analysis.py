#!/usr/bin/env python3
"""
run_v9_error_analysis.py — 误差结构分析专用实验 runner
=====================================================

目的:
  1. 在 v7 统一环境下重跑 seed 1024 (验证 v6->v7 环境差异)
  2. 为 UMC 和 UAMCM 生成 sample-level 数据 (支持联合 (p,u) 分桶)
  3. 同时保存 ubins CSV (向后兼容) 和 npz sample-level 数据
  4. 完整记录实验过程用于论文分析

输出结构:
  ckpt/v9_error_analysis/
    samples/          — sample-level npz (y_true, y_pred_uncalib, y_pred_calib, sigma2)
    ubins/            — per-uncertainty-bin 聚合 CSV
    logs/             — 完整训练日志 (stdout + stderr)
    epochs/           — 每个实验的 per-epoch 训练曲线 CSV
    configs/          — 每个实验的完整配置 JSON (可复现)
    summary/          — 汇总 CSV

用法:
  python run_v9_error_analysis.py --phase p0          # seed 1024 验证
  python run_v9_error_analysis.py --phase p1          # 核心 sample-level (seed 2024)
  python run_v9_error_analysis.py --phase p2          # 多 seed 补全
  python run_v9_error_analysis.py --phase all         # 全部
  python run_v9_error_analysis.py --smoke             # 冒烟测试
  python run_v9_error_analysis.py --verify-backbone   # 验证 backbone
  python run_v9_error_analysis.py --dry-run           # 预览不执行
"""

import argparse
import json
import os
import re
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from time import time

# ── 路径配置 ──────────────────────────────────────────

PAPER_ROOT = "/root/shared-nvme/PAPER"
CKPT_ROOT = os.path.join(PAPER_ROOT, "ckpt")
OUT_DIR = os.path.join(CKPT_ROOT, "v9_error_analysis")
SAMPLE_DIR = os.path.join(OUT_DIR, "samples")
UBINS_DIR = os.path.join(OUT_DIR, "ubins")
LOG_DIR = os.path.join(OUT_DIR, "logs")
EPOCH_DIR = os.path.join(OUT_DIR, "epochs")
CONFIG_DIR = os.path.join(OUT_DIR, "configs")
SUMMARY_DIR = os.path.join(OUT_DIR, "summary")

for d in [OUT_DIR, SAMPLE_DIR, UBINS_DIR, LOG_DIR, EPOCH_DIR, CONFIG_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 导入训练脚本 ──────────────────────────────────────

sys.path.insert(0, os.path.join(PAPER_ROOT, "UMC"))

from train_neu_ali import trial as trial_ali
from train_neu_avazu import trial as trial_avazu

try:
    from train_neu_criteo import trial as trial_criteo
    HAS_CRITEO = True
except ImportError:
    HAS_CRITEO = False

# ── Backbone 配置 (与 v7 完全一致) ────────────────────

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

# ── 实验定义 ──────────────────────────────────────────

def make_experiments(phase, dataset_filter=None, method_filter=None, seed_filter=None):
    experiments = []

    if phase in ("p0", "all"):
        for method in ["umc", "uamcm"]:
            experiments.append({
                "name": f"p0_{method}_s1024",
                "dataset": "aliccp",
                "method": method,
                "calib_seed": 1024,
                "phase": "p0",
            })

    if phase in ("p1", "all"):
        # Avazu 优先: Simpson's Paradox 最关键, 80min 后即可开始分析
        datasets = ["avazu", "criteo", "aliccp"] if HAS_CRITEO else ["avazu", "aliccp"]
        for ds in datasets:
            for method in ["umc", "uamcm"]:
                experiments.append({
                    "name": f"p1_{ds}_{method}_s2024",
                    "dataset": ds,
                    "method": method,
                    "calib_seed": 2024,
                    "phase": "p1",
                })

    if phase in ("p2", "all"):
        datasets = ["aliccp", "avazu", "criteo"] if HAS_CRITEO else ["aliccp", "avazu"]
        for ds in datasets:
            for method in ["umc", "uamcm"]:
                for seed in [1024, 3024]:
                    if phase == "all" and ds == "aliccp" and seed == 1024:
                        continue
                    experiments.append({
                        "name": f"p2_{ds}_{method}_s{seed}",
                        "dataset": ds,
                        "method": method,
                        "calib_seed": seed,
                        "phase": "p2",
                    })

    if dataset_filter:
        experiments = [e for e in experiments if e["dataset"] == dataset_filter]
    if method_filter:
        experiments = [e for e in experiments if e["method"] == method_filter]
    if seed_filter:
        experiments = [e for e in experiments if e["calib_seed"] == seed_filter]

    return experiments


def build_config(exp, smoke=False, run_id=""):
    config = {}
    config.update(BACKBONE_COMMON)
    config.update(CALIB_COMMON)
    config.update(DATASET_CONFIG[exp["dataset"]])
    config["method"] = exp["method"]
    config["calib_seed"] = exp["calib_seed"]

    if smoke:
        config["epochs_calib"] = 1
        config["calib_early_stop"] = False

    run_name = f"{exp['dataset']}_{exp['method']}_s{exp['calib_seed']}"
    # 文件名加时间戳: 防止重跑覆盖, 保留完整历史
    file_tag = f"{run_name}_{run_id}" if run_id else run_name
    config["uncertainty_bin_save_path"] = os.path.join(UBINS_DIR, f"{file_tag}_ubins.csv")
    config["sample_level_save_path"] = os.path.join(SAMPLE_DIR, f"{file_tag}_samples.npz")

    return config, run_name, file_tag


# ── Backbone 验证 ────────────────────────────────────

def verify_backbone():
    """比较所有数据集的 backbone 在 legacy 和 shared-nvme 路径是否一致"""
    print("=" * 60)
    print("BACKBONE VERIFICATION")
    print("=" * 60)

    for ds in ["aliccp", "avazu", "criteo"]:
        ckpt_name = (
            f"data_name={ds}_model_name=packed_deepfm_batch_size=32768_dropout=0.1"
            f"_init_std=0.0001_lr=0.0005_l2_reg=1e-05_seed=1024"
            f"_num_estimators=16_alpha=1.0_gamma=1_.pth"
        )
        legacy_path = f"/root/PAPER/ckpt/{ckpt_name}"
        shared_path = f"/root/shared-nvme/PAPER/ckpt/{ckpt_name}"

        print(f"\n  [{ds}]")
        results = {}
        for label, p in [("legacy", legacy_path), ("shared", shared_path)]:
            if os.path.exists(p):
                size = os.path.getsize(p) / (1024 * 1024)
                h = hashlib.md5()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(8192 * 1024), b""):
                        h.update(chunk)
                md5 = h.hexdigest()
                results[label] = md5
                print(f"    {label}: {size:.1f}MB  md5={md5}")
            else:
                results[label] = None
                print(f"    {label}: NOT FOUND")

        if results.get("legacy") and results.get("shared"):
            match = results["legacy"] == results["shared"]
            print(f"    MATCH: {match}")
        elif results.get("shared"):
            print(f"    (legacy not found, shared exists - OK for v7+ environment)")

    print()


# ── Epoch Trajectory 提取 (复用 v7 逻辑) ─────────────

def extract_epoch_trajectory(log_content):
    """从训练日志中提取 per-epoch 指标, 用于论文 learning-curve 图表"""
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
            # 向后找 ece_track 行
            for j in range(i + 1, min(i + 5, len(lines))):
                m_ece = re.search(
                    r"ece_track epoch=\d+ test_ece=([\d.eE+-]+)", lines[j]
                )
                if m_ece:
                    test_metrics["ece"] = float(m_ece.group(1))
                    break
            # 向后找 test_ 指标行
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


# ── TeeStream (同时捕获 stdout + stderr) ─────────────

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


# ── 主流程 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V9 Error Analysis Runner")
    parser.add_argument("--phase", choices=["p0", "p1", "p2", "all"], default="p1")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Smoke test (1 epoch)")
    parser.add_argument("--verify-backbone", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.verify_backbone:
        verify_backbone()
        return

    experiments = make_experiments(
        args.phase,
        dataset_filter=args.dataset,
        method_filter=args.method,
        seed_filter=args.seed,
    )

    if not experiments:
        print("No experiments match the filter criteria.")
        return

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"V9 Error Analysis Runner")
    print(f"  Run ID: {run_id}")
    print(f"  Phase: {args.phase}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Smoke: {args.smoke}")
    print(f"  Start: {datetime.now().isoformat()}")
    print()

    for i, exp in enumerate(experiments):
        print(f"  [{i+1}/{len(experiments)}] {exp['name']}  "
              f"dataset={exp['dataset']}  method={exp['method']}  "
              f"calib_seed={exp['calib_seed']}")

    if args.dry_run:
        print("\n[DRY RUN] Would run above experiments. Exiting.")
        return

    print()

    # 运行
    results = []
    for i, exp in enumerate(experiments):
        config, run_name, file_tag = build_config(exp, smoke=args.smoke, run_id=run_id)

        # 保存完整配置 (可复现)
        config_path = os.path.join(CONFIG_DIR, f"{file_tag}.json")
        config_record = {
            "run_name": run_name,
            "file_tag": file_tag,
            "run_id": run_id,
            "experiment": exp,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "smoke": args.smoke,
        }
        with open(config_path, "w") as f:
            json.dump(config_record, f, indent=2)

        # 日志: 同时捕获 stdout 和 stderr
        log_path = os.path.join(LOG_DIR, f"{file_tag}.log")
        log_file = open(log_path, "w")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeStream(old_stdout, log_file)
        sys.stderr = TeeStream(old_stderr, log_file)

        print("=" * 60)
        print(f"  [{i+1}/{len(experiments)}] {exp['name']}")
        print(f"  dataset={exp['dataset']}  method={exp['method']}  "
              f"calib_seed={exp['calib_seed']}  phase={exp['phase']}")
        print(f"  Start: {datetime.now().isoformat()}")
        print(f"  Config saved: {config_path}")
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

        # 恢复 stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()

        # 从日志提取 per-epoch trajectory
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

        # 列出本次实验产出的文件
        print(f"  Artifacts:")
        for label, path in [
            ("config", config_path),
            ("log", log_path),
            ("epochs", os.path.join(EPOCH_DIR, f"{file_tag}_epochs.csv")),
            ("ubins", os.path.join(UBINS_DIR, f"{file_tag}_ubins.csv")),
            ("ubins_ece", os.path.join(UBINS_DIR, f"{file_tag}_ubins_ece_best.csv")),
            ("samples", os.path.join(SAMPLE_DIR, f"{file_tag}_samples.npz")),
            ("samples_ece", os.path.join(SAMPLE_DIR, f"{file_tag}_samples_ece_best.npz")),
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

    # 保存汇总
    import pandas as pd
    df = pd.DataFrame(results)
    summary_path = os.path.join(SUMMARY_DIR, f"summary_v9_{args.phase}_{run_id}.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    print(df.to_string(index=False))
    total_time = sum(r.get("elapsed_s", 0) for r in results)
    print(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"End: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
