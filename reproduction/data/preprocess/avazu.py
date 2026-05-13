"""Avazu 预处理 — Kaggle Avazu CTR Prediction 数据集。

输入: data/raw/avazu/train  (Kaggle 解压后)
输出: data/processed/avazu/data.pkl

列结构 (Kaggle train.csv):
    id, click, hour, C1, banner_pos, site_id, site_domain, site_category,
    app_id, app_domain, app_category, device_id, device_ip, device_model,
    device_type, device_conn_type, C14, C15, C16, C17, C18, C19, C20, C21

预处理（参考 UMC/dataset/avazu_process.ipynb）:
    1. 去掉前 3 列 (id, click, hour) 中的 id 和 hour，保留 click 作 label
    2. 其余 21 列作稀疏特征（OrdinalEncoder）
    3. 论文 field_index=2 指向 banner_pos 字段（u 信号挂钩位置）

实际 sparse_features 列数 = 24 - 3 (id, click, hour) = 21；但 plan §C.4 说
Avazu sparse_feature_count=24（含 hour 等被 encode 的 3 列），所以预处理
保留 hour + C1 + banner_pos 作为前 3 列（field_index=2 指 banner_pos）。

CLI:
    python -m reproduction.data.preprocess.avazu
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from ._common import resolve_paths, encode_sparse_features, write_pkl_and_meta


# Avazu sparse features（与原 ipynb 一致：去掉 id，保留 click 作 label）
AVAZU_SPARSE_FEATURES: List[str] = [
    "hour", "C1", "banner_pos",                  # field_index=2 指 banner_pos
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
FIELD_INDEX_AVAZU = 2                            # 与 train_neu_avazu.py L66 一致


def main() -> int:
    import pandas as pd

    raw_dir, processed_dir = resolve_paths("avazu")
    raw_csv = raw_dir / "train"                  # uncompressed Kaggle train

    if not raw_csv.exists():
        print(f"[avazu] ERROR: {raw_csv} not found. Run `download.py` first.")
        return 1

    print(f"[avazu] reading {raw_csv}")
    df = pd.read_csv(raw_csv)
    print(f"[avazu] raw shape: {df.shape}")
    print(f"[avazu] columns: {list(df.columns)}")

    # 保留稀疏特征 + click
    missing = [c for c in AVAZU_SPARSE_FEATURES if c not in df.columns]
    if missing:
        print(f"[avazu] ERROR: missing columns: {missing}")
        return 1

    encoded, vocab = encode_sparse_features(
        df[AVAZU_SPARSE_FEATURES + ["click"]],
        sparse_features=AVAZU_SPARSE_FEATURES,
        label_col="click",
        encoder="ordinal",
    )
    write_pkl_and_meta(
        encoded,
        processed_dir,
        sparse_features=AVAZU_SPARSE_FEATURES,
        vocab_sizes=vocab,
        field_index=FIELD_INDEX_AVAZU,
        label_col="click",
    )
    print(f"[avazu] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
