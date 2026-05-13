"""reproduction.data.download — 数据下载与 md5 校验。

CLI 用法:
    # 列出数据集计划
    python -m reproduction.data.download --list

    # 下载所有
    python -m reproduction.data.download --all

    # 下载单个
    python -m reproduction.data.download --dataset aliccp

    # 仅校验已下载文件
    python -m reproduction.data.download --verify-only

数据源（详见 data/README.md）:
    AliCCP: USTC mirror（主）+ Tianchi（备）
    Avazu:  Kaggle（主）+ USTC（备）
    Criteo: Criteo 1TB（主）+ Kaggle（备）

USTC 分享链接 (rec.ustc.edu.cn) 因 OAuth 保护无法直接 aria2c；本脚本生成
**人工操作指南**（含密码、Kaggle CLI 命令），并自动校验已落地文件的 md5。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


_HERE = Path(__file__).resolve().parent           # 30_reproduction/reproduction/data/
_PROJECT_ROOT = _HERE.parent.parent               # 30_reproduction/


def _data_root() -> Path:
    """从 UMC/_paths.py 解析 DATA_ROOT 的父级（data/）。"""
    sys.path.insert(0, str(_PROJECT_ROOT / "UMC"))
    from _paths import DATA_ROOT                  # type: ignore

    return Path(DATA_ROOT).parent                 # data/processed 的父 = data/


# ============================================================================
# 数据源声明（与 data/README.md 对齐）
# ============================================================================

DATASETS: Dict[str, Dict[str, Any]] = {
    "aliccp": {
        "primary": {
            "type": "ustc_share",
            "url": "https://rec.ustc.edu.cn/share/5a70c6c0-9e4a-11ef-af55-8dfb3f6b3310",
            "password": "5277",
            "note": "USTC 分享链接需手动登录下载，含 4 个 CSV: common_features_{train,test}.csv + sample_skeleton_{train,test}.csv",
        },
        "backup": {
            "type": "tianchi",
            "url": "https://tianchi.aliyun.com/dataset/408",
            "note": "天池数据集，需登录 Aliyun 账号",
        },
        "expected_files": [
            "common_features_train.csv",
            "common_features_test.csv",
            "sample_skeleton_train.csv",
            "sample_skeleton_test.csv",
        ],
        "size_gb": 15,
    },
    "avazu": {
        "primary": {
            "type": "kaggle",
            "competition": "avazu-ctr-prediction",
            "file": "train.gz",
            "cli": "kaggle competitions download -c avazu-ctr-prediction -f train.gz",
        },
        "backup": {
            "type": "ustc_share",
            "url": "https://rec.ustc.edu.cn/share/5a70c6c0-9e4a-11ef-af55-8dfb3f6b3310",
            "password": "5277",
            "note": "USTC 同一分享包含 Avazu train 文件",
        },
        "expected_files": ["train"],   # uncompressed
        "size_gb": 5,
    },
    "criteo": {
        "primary": {
            "type": "criteo_1tb",
            "url": "https://ailab.criteo.com/criteo-1tb-click-logs-dataset/",
            "note": "Criteo 1TB Click Logs，下载前 5GB 即可（论文用前 N 天切片）",
        },
        "backup": {
            "type": "kaggle",
            "competition": "criteo-display-ad-challenge",
            "file": "dac.tar.gz",
            "cli": "kaggle competitions download -c criteo-display-ad-challenge -f dac.tar.gz",
        },
        "expected_files": ["train.txt"],
        "size_gb": 12,
    },
}


# ============================================================================
# md5 校验
# ============================================================================

def md5sum(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """计算文件 md5（流式，适用大文件）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(dataset: str, raw_root: Path) -> Dict[str, Any]:
    """检查某数据集预期文件是否齐全 + 计算 md5。"""
    meta = DATASETS[dataset]
    ddir = raw_root / dataset
    report: Dict[str, Any] = {
        "dataset": dataset,
        "raw_dir": str(ddir),
        "files": {},
        "missing": [],
        "complete": False,
    }
    if not ddir.exists():
        report["missing"] = list(meta["expected_files"])
        return report
    for fname in meta["expected_files"]:
        fpath = ddir / fname
        if not fpath.exists():
            report["missing"].append(fname)
            continue
        report["files"][fname] = {
            "size_mb": round(fpath.stat().st_size / 1024**2, 1),
            "md5": md5sum(fpath),
        }
    report["complete"] = not report["missing"]
    return report


# ============================================================================
# 下载（人工指南）
# ============================================================================

def print_download_guide(dataset: str) -> None:
    """打印某数据集的下载指南（含 USTC 密码 / Kaggle CLI 命令）。"""
    meta = DATASETS[dataset]
    print(f"\n=========== Dataset: {dataset} ===========")
    print(f"Expected size: ~{meta['size_gb']} GB")
    print(f"Expected files: {meta['expected_files']}")

    print(f"\n--- Primary source ---")
    p = meta["primary"]
    if p["type"] == "ustc_share":
        print(f"  USTC share URL: {p['url']}")
        print(f"  Password: {p['password']}")
        print(f"  Action: 浏览器打开 → 输入密码 → 下载 → 解压到 data/raw/{dataset}/")
        print(f"  Note: {p.get('note', '')}")
    elif p["type"] == "kaggle":
        print(f"  Kaggle CLI: {p['cli']}")
        print(f"  Action: cd data/raw/{dataset} && {p['cli']} && gunzip {p['file']}")
        print(f"  Prerequisite: kaggle CLI installed + ~/.kaggle/kaggle.json configured")
    elif p["type"] == "criteo_1tb":
        print(f"  URL: {p['url']}")
        print(f"  Note: {p['note']}")
        print(f"  Action: 手动下载前 5GB → data/raw/{dataset}/")

    print(f"\n--- Backup source ---")
    b = meta["backup"]
    if b["type"] == "ustc_share":
        print(f"  USTC: {b['url']} (password: {b['password']})")
    elif b["type"] == "kaggle":
        print(f"  Kaggle CLI: {b['cli']}")
    elif b["type"] == "tianchi":
        print(f"  Tianchi: {b['url']} ({b.get('note', '')})")


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reproduction.data.download")
    p.add_argument("--list", action="store_true", help="列出全部数据集计划")
    p.add_argument("--all", action="store_true", help="为所有数据集打印下载指南")
    p.add_argument("--dataset", choices=list(DATASETS.keys()),
                   help="只处理单个数据集")
    p.add_argument("--verify-only", action="store_true",
                   help="仅校验已落地文件的 md5，不打印下载指南")
    p.add_argument("--write-manifest", type=str, default=None,
                   help="把 md5 + size 写入指定 manifest 文件路径")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    raw_root = _data_root() / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    if args.list:
        for name, meta in DATASETS.items():
            print(f"  {name:10s}  ~{meta['size_gb']:3d} GB  files={meta['expected_files']}")
        return 0

    targets: List[str] = [args.dataset] if args.dataset else list(DATASETS.keys())

    reports: Dict[str, Any] = {}
    for ds in targets:
        rpt = verify_dataset(ds, raw_root)
        reports[ds] = rpt
        complete = "OK" if rpt["complete"] else f"MISSING({len(rpt['missing'])})"
        print(f"\n[{ds}] status: {complete}")
        for fname, info in rpt["files"].items():
            print(f"  {fname:40s}  {info['size_mb']:>8.1f} MB  md5={info['md5']}")
        for fname in rpt["missing"]:
            print(f"  MISSING: {fname}")

        if not rpt["complete"] and not args.verify_only:
            print_download_guide(ds)

    if args.write_manifest:
        mpath = Path(args.write_manifest)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(reports, indent=2, ensure_ascii=False))
        print(f"\nManifest written: {mpath}")

    return 0 if all(r["complete"] for r in reports.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
