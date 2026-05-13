"""Criteo 预处理 — Criteo Display Advertising Challenge / 1TB Click Logs。

输入:  data/raw/criteo/train.txt  (Criteo 标准 TSV 格式)
输出:  data/processed/criteo/data.pkl

Criteo 标准格式（TSV，无 header）:
    label  I1 I2 ... I13  C1 C2 ... C26
    - I1..I13: 整数特征（13 个 numerical）
    - C1..C26: 类别特征（26 个 categorical，hex 编码）
    - label: 0/1 click

论文使用前 N 天数据（约 50M 样本）；本脚本默认处理整个 train.txt。
如需切片，先 `head -50000000 train.txt > train_50m.txt`。

注意:
    - UMC 论文 Criteo field_index=23（指 C10，u 信号挂钩位置；与
      train_neu_criteo.py L66 一致）
    - 39 sparse features = 13 (I1-I13 离散化) + 26 (C1-C26)
    - 整数特征 I_i 先 bucketize 再 encode（log(I+1) → int），与原 UMC 实现一致

CLI:
    python -m reproduction.data.preprocess.criteo
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ._common import resolve_paths, encode_sparse_features, write_pkl_and_meta


# Criteo 标准列名
INTEGER_FEATURES: List[str] = [f"I{i}" for i in range(1, 14)]   # 13
CATEGORICAL_FEATURES: List[str] = [f"C{i}" for i in range(1, 27)]   # 26
ALL_FEATURES: List[str] = INTEGER_FEATURES + CATEGORICAL_FEATURES   # 39
FIELD_INDEX_CRITEO = 23   # 与 train_neu_criteo.py L66 一致（C11）


def bucketize_integer(col: pd.Series) -> pd.Series:
    """整数特征离散化：x → floor(log2(x+1))。原 UMC 标准做法。"""
    # 填 NaN 为 0
    x = col.fillna(0).astype(float).clip(lower=0)
    return np.floor(np.log2(x + 1)).astype(int).clip(upper=100)


def main() -> int:
    raw_dir, processed_dir = resolve_paths("criteo")
    raw_tsv = raw_dir / "train.txt"

    if not raw_tsv.exists():
        # 兼容 dac.txt / train_50m.txt
        for alt in ["dac.txt", "train_50m.txt", "data.txt"]:
            p = raw_dir / alt
            if p.exists():
                raw_tsv = p
                print(f"[criteo] using alt file: {raw_tsv}")
                break
        else:
            print(f"[criteo] ERROR: {raw_tsv} not found. Run `download.py` first.")
            return 1

    print(f"[criteo] reading {raw_tsv}")
    column_names = ["click"] + ALL_FEATURES
    df = pd.read_csv(
        raw_tsv,
        sep="\t",
        header=None,
        names=column_names,
        dtype={col: "object" for col in CATEGORICAL_FEATURES},
        na_values=["", "\\N"],
    )
    print(f"[criteo] raw shape: {df.shape}")

    # 整数特征 bucketize
    print(f"[criteo] bucketizing integer features (log2)")
    for col in INTEGER_FEATURES:
        df[col] = bucketize_integer(df[col])

    # 类别特征 fillna
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("__missing__").astype(str)

    encoded, vocab = encode_sparse_features(
        df[ALL_FEATURES + ["click"]],
        sparse_features=ALL_FEATURES,
        label_col="click",
        encoder="ordinal",
    )
    write_pkl_and_meta(
        encoded,
        processed_dir,
        sparse_features=ALL_FEATURES,
        vocab_sizes=vocab,
        field_index=FIELD_INDEX_CRITEO,
        label_col="click",
    )
    print(f"[criteo] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
