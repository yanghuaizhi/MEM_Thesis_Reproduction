"""Avazu 预处理 — Kaggle Avazu CTR Prediction 数据集。

输入: data/raw/avazu/train  (Kaggle 解压后)
输出: data/processed/avazu/data.pkl

列结构 (Kaggle train.csv):
    id, click, hour, C1, banner_pos, site_id, site_domain, site_category,
    app_id, app_domain, app_category, device_id, device_ip, device_model,
    device_type, device_conn_type, C14, C15, C16, C17, C18, C19, C20, C21

预处理（严格对齐 UMC/dataset/avazu_process.ipynb Cell 7）:
    sparse_features = list(avazu_data.columns)[3:]   # 切片从 C1 开始
    # 即 21 列稀疏，去掉 id/click/hour 前 3 列

    OrdinalEncoder().fit_transform(avazu_data[sparse_features])
    save_data = avazu_data[sparse_features + ['click']]

field_index=2 在 ipynb 列序 [C1, banner_pos, site_id, ...] 中 = **site_id**
（u 信号挂钩字段；与 train_neu_avazu.py L66 一致）。

CLI:
    python -m reproduction.data.preprocess.avazu
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from ._common import resolve_paths, encode_sparse_features, write_pkl_and_meta


# Avazu sparse features（严格对齐 ipynb Cell 7: columns[3:]，去掉 id/click/hour）
# 列序: C1, banner_pos, site_id (index=2), site_domain, site_category, ...
AVAZU_SPARSE_FEATURES: List[str] = [
    "C1", "banner_pos",
    "site_id", "site_domain", "site_category",        # site_id 在 index=2（field_index 指此）
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
# 21 columns total
assert len(AVAZU_SPARSE_FEATURES) == 21, f"expected 21, got {len(AVAZU_SPARSE_FEATURES)}"

FIELD_INDEX_AVAZU = 2                            # 指向 site_id（train_neu_avazu.py L66 一致）
FIELD_NAME_AVAZU = "site_id"                     # u 信号挂钩特征


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
