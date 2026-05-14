"""reproduction.orchestrator — 统一编排器。

CLI 用法:
    # 阶段一：PackedDeepFM backbone（9 个）
    python -m reproduction.orchestrator --stage pretrain --resume

    # 主实验（99 任务）— 全部
    python -m reproduction.orchestrator --stage main --resume

    # 主实验 — 子集
    python -m reproduction.orchestrator --stage main --dataset aliccp --method uamcm --resume
    python -m reproduction.orchestrator --stage main --method-type statistical --resume

    # v9 sample-level inference（依赖 main 完成）
    python -m reproduction.orchestrator --stage v9 --resume

    # v10 u_mode 消融
    python -m reproduction.orchestrator --stage v10 --resume

    # 只列任务，不执行
    python -m reproduction.orchestrator --stage main --dry-run

设计:
    - 读 configs/{datasets,methods,experiments,hardware}/*.yaml
    - 合并出每个 run 的 effective config dict
    - 写 experiments/runs/{stage}/{dataset}/{method}/seed_{N}/run_config.json
    - subprocess 启 `python -m reproduction._runner --config <path>`
    - _runner 写 done.flag 标记完成；--resume 时跳过已有 done.flag 的 run
    - status.json 每个 run 前后更新（供 ssh + jq 远程诊断）

不做（保持 minimal）:
    - 并行调度（单 GPU 串行；如需并发，外层 `&` + wait）
    - 日志聚合（每个 run 的 stdout 单独写 train.log）
    - 失败重试（失败后 manual review，再次 --resume 即可）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ============================================================================
# Paths
# ============================================================================

_HERE = Path(__file__).resolve().parent              # 30_reproduction/reproduction/
_PROJECT_ROOT = _HERE.parent                          # 30_reproduction/
_CONFIGS_DIR = _HERE / "configs"


def _experiments_root() -> Path:
    """从 UMC/_paths.py 读 CKPT_ROOT，作为 experiments/runs 的父目录。"""
    sys.path.insert(0, str(_PROJECT_ROOT / "UMC"))
    from _paths import CKPT_ROOT  # type: ignore

    return Path(CKPT_ROOT)


# ============================================================================
# Config loading
# ============================================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_configs() -> Dict[str, Dict[str, Any]]:
    """加载所有 YAML，返回 {category: {name: cfg_dict}} 结构。"""
    out: Dict[str, Dict[str, Any]] = {
        "datasets": {},
        "methods": {},
        "experiments": {},
        "hardware": {},
    }
    for category in out.keys():
        category_dir = _CONFIGS_DIR / category
        if not category_dir.exists():
            continue
        for f in sorted(category_dir.glob("*.yaml")):
            cfg = _load_yaml(f)
            out[category][f.stem] = cfg
    out["paths"] = _load_yaml(_CONFIGS_DIR / "paths.yaml")
    return out


# ============================================================================
# Run plan generation
# ============================================================================

def _build_pretrain_plan(configs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成 3 个 backbone 任务（3 datasets × 1 fixed seed=1024）。

    设计:
        - backbone checkpoint 文件名编码了 seed（train_neu_*.py L196-211
          的 experiment_name），calib 阶段永远加载 seed=1024 的 backbone
          （`_build_calib_config_update` 传 `seed: backbone["pretrain_seed"]`）。
        - 因此跑 seed=2024/3024 的 backbone 没有意义（calib 不会用）。
        - 校准 3 seeds 仍在 calib 阶段通过 `calib_seed` 实现（共用同一 backbone）。
    """
    main = configs["experiments"]["main_99"]
    backbone = main["backbone"]
    pretrain_seed = int(backbone["pretrain_seed"])    # 固定 1024
    runs: List[Dict[str, Any]] = []
    for dataset in main["datasets"]:
        ds = configs["datasets"][dataset]
        cu = {
            "data_name": dataset,
            "model_name": backbone["model_name"],
            "batch_size": ds["batch_size"]["pretrain"],
            "dropout": backbone["dropout"],
            "init_std": backbone["init_std"],
            "lr": backbone["lr_pretrain"],
            "l2_reg": backbone["l2_reg"],
            "embedding_dim": backbone["embedding_dim"],
            "num_estimators": backbone["num_estimators"],  # 16
            "alpha": backbone["alpha"],
            "gamma": backbone["gamma"],
            "seed": pretrain_seed,                    # 固定 1024
            "epochs": backbone["epochs"],
            "patience": backbone["patience"],
            "monitor": backbone["monitor"],
            "mode": backbone["mode"],
            "num_workers": configs["hardware"]["rtx5090"]["dataloader"]["num_workers"],
            "pin_memory": configs["hardware"]["rtx5090"]["dataloader"]["pin_memory"],
            "persistent_workers": configs["hardware"]["rtx5090"]["dataloader"]["persistent_workers"],
            "prefetch_factor": configs["hardware"]["rtx5090"]["dataloader"].get("prefetch_factor"),  # H1 fix
        }
        runs.append(
            {
                "stage": "pretrain",
                "entry": "pretrain",
                "dataset": dataset,
                "method": "_backbone",
                "seed": pretrain_seed,
                "config_update": cu,
            }
        )
    return runs


def _build_main_plan(
    configs: Dict[str, Any],
    filter_dataset: Optional[str],
    filter_method: Optional[str],
    filter_method_type: Optional[str],
) -> List[Dict[str, Any]]:
    """生成 99 校准任务（11 methods × 3 datasets × 3 seeds）。"""
    main = configs["experiments"]["main_99"]
    hw = configs["hardware"]["rtx5090"]
    runs: List[Dict[str, Any]] = []
    for dataset in main["datasets"]:
        if filter_dataset and dataset != filter_dataset:
            continue
        ds = configs["datasets"][dataset]
        for method in main["methods"]:
            if filter_method and method != filter_method:
                continue
            mcfg = configs["methods"][method]
            if filter_method_type and mcfg["type"] != filter_method_type:
                continue
            for seed in main["seeds"]:
                cu = _build_calib_config_update(
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    ds=ds,
                    mcfg=mcfg,
                    main=main,
                    hw=hw,
                )
                # FIX-5: 注入 uncertainty_bin_save_path 让 csv 落盘
                run_dir = _run_dir("main", dataset, method, seed)
                cu["uncertainty_bin_save_path"] = str(run_dir / "uncertainty_bins.csv")
                runs.append(
                    {
                        "stage": "main",
                        "entry": mcfg["entry"],
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "config_update": cu,
                    }
                )
    return runs


def _build_calib_config_update(
    dataset: str,
    method: str,
    seed: int,
    ds: Dict[str, Any],
    mcfg: Dict[str, Any],
    main: Dict[str, Any],
    hw: Dict[str, Any],
) -> Dict[str, Any]:
    """合并 dataset + method + main + hardware → 校准 config_update。"""
    backbone = main["backbone"]
    cu: Dict[str, Any] = {
        # 数据集
        "data_name": dataset,
        "field_index": ds["field_index"],
        "batch_size": ds["batch_size"]["pretrain"],          # backbone load 用
        "batch_size_calib": ds["batch_size"]["calib"],
        # backbone（calib 依赖 backbone 文件名编码）
        "model_name": backbone["model_name"],
        "num_estimators": backbone["num_estimators"],
        "dropout": backbone["dropout"],
        "init_std": backbone["init_std"],
        "lr": backbone["lr_pretrain"],
        "l2_reg": backbone["l2_reg"],
        "alpha": backbone["alpha"],
        "gamma": backbone["gamma"],
        "seed": backbone["pretrain_seed"],                   # backbone seed=1024 固定
        # 校准
        "calib_seed": seed,                                  # 校准的 3 个 seed
        "method": method,
        # 评估
        "ece_M": main["evaluation"]["ece_bins"],
        "uncertainty_bin_eval": main["evaluation"]["uncertainty_bin_eval"],
        "uncertainty_bin_num_bins": main["evaluation"]["uncertainty_bin_num_bins"],
        "uncertainty_bin_ece_M": main["evaluation"]["ece_bins"],
        # DataLoader（Tier 1）
        "num_workers": hw["dataloader"]["num_workers"],
        "pin_memory": hw["dataloader"]["pin_memory"],
        "persistent_workers": hw["dataloader"]["persistent_workers"],
        "prefetch_factor": hw["dataloader"].get("prefetch_factor"),    # H1 fix: 注入
    }
    # 方法超参（仅神经方法有 calib hyperparameters）
    if mcfg["type"] != "statistical":
        hp = mcfg["hyperparameters"]
        cu.update(
            {
                "lr_calib": hp["lr_calib"],
                "epochs_calib": hp["epochs_calib"],
                "calib_early_stop": True,
                "calib_patience": hp["patience"],
                "calib_min_delta": hp["min_delta"],
                "calib_restore_best": hp["restore_best"],
                "calib_log_every": hp["calib_log_every"],
            }
        )
        # u 相关字段（仅 uses_u=True 方法）
        if mcfg["uses_u"]:
            for k in ("u_mode", "u_use_norm", "u_clip_min", "u_clip_max",
                      "u_use_resid", "u_resid_bins"):
                if k in hp:
                    cu[k] = hp[k]
            # UAMCM 特定（integral_dim 已从 YAML 移除：UAMCM 构造函数无此参数）
            for k in ("u_min", "u_max",
                      "alpha_max", "delta_scale_init"):
                if k in hp:
                    cu[k] = hp[k]
        # SCL 共用
        for k in ("scl_lam", "scl_beta"):
            if k in hp:
                cu[k] = hp[k]
    return cu


def _build_v9_plan(configs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v9 sample-level inference: 复用 train_neu 流程 + 注入 sample_level_save_path。

    FIX-9: 跑 3 seed × 3 dataset × 2 method (umc + uamcm) = 18 任务
    设计原理: run_v9 不需新写 inference 逻辑——直接调 train_neu trial，UMC 内部
    L860-864/L943-948 会自动调 utils.save_samples.save_sample_level() 保存 NPZ。
    """
    v9 = configs["experiments"]["v9_error_analysis"]
    hw = configs["hardware"]["rtx5090"]
    main = configs["experiments"]["main_99"]
    runs: List[Dict[str, Any]] = []
    samples_dir = v9["output"]["samples_dir"]      # 相对 CKPT_ROOT
    exp_root = _experiments_root()

    for dataset in v9["datasets"]:
        ds = configs["datasets"][dataset]
        for method in v9["inference"]["methods"]:    # B4: umc + uamcm 双方法
            mcfg = configs["methods"][method]
            for seed in v9["seeds"]:                # FIX-9: 3 seeds
                cu = _build_calib_config_update(
                    dataset=dataset, method=method, seed=seed,
                    ds=ds, mcfg=mcfg, main=main, hw=hw,
                )
                # B5: 注入 sample_level_save_path 让 NPZ 保存
                npz_path = exp_root / samples_dir / dataset / f"{method}_seed_{seed}_samples.npz"
                cu["sample_level_save_path"] = str(npz_path)
                runs.append(
                    {
                        "stage": "v9",
                        "entry": mcfg["entry"],         # train_neu
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "config_update": cu,
                    }
                )
    return runs


def _build_v10_plan(configs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v10 u_mode 消融：固定 UAMCM，仅变 u_mode。"""
    v10 = configs["experiments"]["v10_ablation"]
    hw = configs["hardware"]["rtx5090"]
    main = configs["experiments"]["main_99"]
    runs: List[Dict[str, Any]] = []
    for dataset in v10["datasets"]:
        ds = configs["datasets"][dataset]
        for method in v10["methods"]:               # ["uamcm"]
            mcfg = configs["methods"][method]
            for u_mode in v10["u_modes"]:
                for seed in v10["seeds"]:
                    cu = _build_calib_config_update(
                        dataset=dataset, method=method, seed=seed,
                        ds=ds, mcfg=mcfg, main=main, hw=hw,
                    )
                    cu["u_mode"] = u_mode           # 覆盖
                    # FIX-5: 同 main，注入 uncertainty_bin_save_path
                    method_dir_name = f"{method}_umode_{u_mode}"
                    run_dir = _run_dir("v10", dataset, method_dir_name, seed)
                    cu["uncertainty_bin_save_path"] = str(run_dir / "uncertainty_bins.csv")
                    runs.append(
                        {
                            "stage": "v10",
                            "entry": mcfg["entry"],
                            "dataset": dataset,
                            "method": method_dir_name,
                            "seed": seed,
                            "config_update": cu,
                        }
                    )
    return runs


# ============================================================================
# Run launching
# ============================================================================

def _run_dir(stage: str, dataset: str, method: str, seed: int) -> Path:
    return _experiments_root() / "runs" / stage / dataset / method / f"seed_{seed}"


def _is_done(rdir: Path) -> bool:
    return (rdir / "done.flag").exists()


def _launch_run(
    run: Dict[str, Any],
    hw_cfg: Dict[str, Any],
    dry_run: bool = False,
) -> bool:
    """启动一个 run 的 subprocess。返回 True 表示成功（或 dry-run）。"""
    rdir = _run_dir(run["stage"], run["dataset"], run["method"], run["seed"])
    rdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "entry": run["entry"],
        "dataset": run["dataset"],
        "config_update": run["config_update"],
        "run_dir": str(rdir),
        "tier": {
            "dataloader": hw_cfg["dataloader"],
            "eval": hw_cfg["eval"],
            "precision": hw_cfg["precision"],
            "compile": hw_cfg["compile"],
        },
        "log_path": str(rdir / "train.log"),
    }
    cfg_path = rdir / "run_config.json"
    cfg_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    if dry_run:
        print(f"[dry-run] {run['stage']}/{run['dataset']}/{run['method']}/seed_{run['seed']}")
        return True

    log_fp = open(rdir / "train.log", "w", encoding="utf-8")
    cmd = [
        sys.executable, "-m", "reproduction._runner",
        "--config", str(cfg_path),
    ]
    print(f"[orchestrator] launching: {' '.join(cmd)}")
    started = time.time()
    try:
        ret = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            check=False,
        )
    finally:
        log_fp.close()
    elapsed = time.time() - started
    ok = ret.returncode == 0 and _is_done(rdir)
    status = "OK" if ok else f"FAIL(code={ret.returncode})"
    print(f"[orchestrator] {status} elapsed={elapsed:.1f}s  rdir={rdir}")
    return ok


# ============================================================================
# CLI
# ============================================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reproduction.orchestrator")
    p.add_argument(
        "--stage",
        required=True,
        choices=["pretrain", "main", "v9", "v10"],
        help="实验阶段",
    )
    p.add_argument("--dataset", choices=["aliccp", "avazu", "criteo"],
                   help="只跑某个数据集（仅 main/v10）")
    p.add_argument("--method", help="只跑某个方法（仅 main/v10）")
    p.add_argument("--method-type",
                   choices=["statistical", "neural_baseline", "paper_core"],
                   help="只跑某类方法（仅 main）")
    p.add_argument("--seed", type=int, choices=[1024, 2024, 3024],
                   help="只跑某个 seed")
    p.add_argument("--resume", action="store_true",
                   help="跳过已有 done.flag 的 run")
    p.add_argument("--dry-run", action="store_true",
                   help="仅列任务，不启 subprocess")
    p.add_argument("--max-runs", type=int, default=None,
                   help="最多跑 N 个（调试用）")
    p.add_argument("--parallel", type=int, default=1,
                   help="并发任务数 (默认 1 = 串行；5090 32GB 显存可承载 3-5 并发，"
                        "单 task 仅 ~3GB 显存)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    configs = _load_configs()
    hw_cfg = configs["hardware"]["rtx5090"]

    if args.stage == "pretrain":
        runs = _build_pretrain_plan(configs)
    elif args.stage == "main":
        runs = _build_main_plan(
            configs,
            filter_dataset=args.dataset,
            filter_method=args.method,
            filter_method_type=args.method_type,
        )
    elif args.stage == "v9":
        runs = _build_v9_plan(configs)
    elif args.stage == "v10":
        runs = _build_v10_plan(configs)
    else:
        raise ValueError(args.stage)

    # seed filter
    if args.seed is not None:
        runs = [r for r in runs if r["seed"] == args.seed]

    print(f"[orchestrator] stage={args.stage} total_runs={len(runs)}")

    # resume filter
    pending: List[Dict[str, Any]] = []
    skipped = 0
    for r in runs:
        rdir = _run_dir(r["stage"], r["dataset"], r["method"], r["seed"])
        if args.resume and _is_done(rdir):
            skipped += 1
            continue
        pending.append(r)
    print(f"[orchestrator] pending={len(pending)} skipped(done)={skipped}")

    if args.max_runs:
        pending = pending[: args.max_runs]
        print(f"[orchestrator] max-runs cap: limit to {len(pending)}")

    if args.dry_run:
        for r in pending:
            tag = f"{r['stage']}/{r['dataset']}/{r['method']}/seed_{r['seed']}"
            print(f"  PLANNED  {tag}  entry={r['entry']}")
        return 0

    failed = 0
    if args.parallel > 1 and len(pending) > 1:
        # 并发模式：用 ThreadPoolExecutor 同时启动 N 个 subprocess
        # 每个 task 独立进程 + 独立 CUDA context，共享 GPU 但显存隔离
        # 数值无影响（每个 task 内 setup_seed 独立）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_par = min(args.parallel, len(pending))
        print(f"\n[orchestrator] PARALLEL mode: {n_par} concurrent subprocesses on GPU")
        print(f"[orchestrator] WARN: each task ~3GB GPU mem + ~500MB CUDA ctx; "
              f"keep parallel ≤ 5 on 32GB GPU")
        completed = 0
        with ThreadPoolExecutor(max_workers=n_par) as ex:
            futures = {
                ex.submit(_launch_run, r, hw_cfg, False): r for r in pending
            }
            for fut in as_completed(futures):
                r = futures[fut]
                completed += 1
                try:
                    ok = fut.result()
                except Exception as e:
                    print(f"[orchestrator] EXC in {r}: {e}")
                    ok = False
                tag = f"{r['stage']}/{r['dataset']}/{r['method']}/seed_{r['seed']}"
                status = "OK" if ok else "FAIL"
                print(f"=== [{completed}/{len(pending)}] {status}: {tag} ===")
                if not ok:
                    failed += 1
        print(f"\n[orchestrator] parallel done: {len(pending) - failed}/{len(pending)} OK, {failed} failed")
        return 1 if failed else 0

    # 串行模式（默认）
    for i, r in enumerate(pending, start=1):
        print(f"\n=== [{i}/{len(pending)}] {r['stage']}/{r['dataset']}/{r['method']}/seed_{r['seed']} ===")
        ok = _launch_run(r, hw_cfg=hw_cfg, dry_run=False)
        if not ok:
            failed += 1
            print(f"[orchestrator] STOP on failure (run {i} failed)")
            break

    print(f"\n[orchestrator] done: {len(pending) - failed}/{len(pending)} succeeded, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
