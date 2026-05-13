"""Criteo 预处理 — 严格迁移自 archive ground truth 实现。

ground truth: /Users/y/Research_MEM/10_research_archive/dataset/criteo/preprocess_criteo.py
（论文 v1.13 实验实际使用，签名命名 signed_log1p_square_tokens）

输入: data/raw/criteo/train.txt  (Criteo 标准 TSV: label + I1..I13 + C1..C26)
输出: data/processed/criteo/data.pkl + data_meta.json + feature_summary.csv

预处理算法（archive 真值）:
    数值特征 I1-I13:
        NaN          → "__MISSING__"
        x < 0        → "NEG_{floor(log1p(-x)^2)}"
        x == 0       → "ZERO"
        x > 0        → "POS_{floor(log1p(x)^2)}"
        然后 LabelEncoder 整体 fit_transform
    类别特征 C1-C26:
        rare-merging: 出现 <min_count(=10) 次 → __rare__ (code=1)
        缺失值 → __missing__ (code=0)
        其余 → code 2,3,4,...
    列序:  [I1..I13, C1..C26, label]  (39 sparse + label)
    label 列名 'label' (与 archive 一致)；UMC get_data() 用 columns[-1] 不挑名字

CLI:
    python -m reproduction.data.preprocess.criteo [--min-count 10]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ._common import resolve_paths


# Criteo 标准列定义（与 archive 一致）
NUM_COLS: List[str] = [f"I{i}" for i in range(1, 14)]      # 13 numeric
CAT_COLS: List[str] = [f"C{i}" for i in range(1, 27)]      # 26 categorical
ALL_FEATURES: List[str] = NUM_COLS + CAT_COLS              # 39 sparse
ALL_COLS_INPUT: List[str] = ["label"] + NUM_COLS + CAT_COLS  # 输入 TSV 列序（label 在首）
FIELD_INDEX_CRITEO = 23                                    # 与 train_neu_criteo.py L66 一致
# 列序 [I1..I13, C1..C26]，index=23 = (23 - 13 + 1) = C11
FIELD_NAME_CRITEO = "C11"


def encode_numeric_column_vectorized(series: pd.Series) -> Tuple[np.ndarray, int]:
    """向量化数值特征编码（archive 真值实现）。

    映射到 (sign, bucket) compound key，再 LabelEncoder。
    与 scalar `bucket_numeric_token` 产生相同 token 集合（仅 LE id 顺序差异，
    对从头训练的 embedding 表无影响）。

    Returns: (encoded_int32_array, n_unique_tokens)
    """
    values = series.values.astype(np.float64)
    is_missing = np.isnan(values)
    safe = np.where(is_missing, 0.0, values)

    bucket = np.floor(np.log1p(np.abs(safe)) ** 2).astype(np.int32)
    max_bucket = int(bucket.max()) + 1

    # sign: 0=MISSING, 1=ZERO, 2=NEG, 3=POS
    sign = np.where(
        is_missing, 0,
        np.where(safe < 0, 2, np.where(safe > 0, 3, 1)),
    ).astype(np.int32)

    compound = sign * max_bucket + bucket
    le = LabelEncoder()
    encoded = le.fit_transform(compound).astype(np.int32)
    return encoded, len(le.classes_)


def encode_categorical_with_rare(cat_series: pd.Series, min_count: int = 10) -> Tuple[np.ndarray, int, float]:
    """类别特征 rare-merging 编码（archive 真值实现）。

    vocab 布局: 0=__missing__, 1=__rare__, 2..N=normal categories
    """
    cat_codes = cat_series.cat.codes.values
    n_raw_unique = len(cat_series.cat.categories)

    valid = cat_codes[cat_codes >= 0]
    code_counts = np.bincount(valid, minlength=n_raw_unique)

    rare_mask = code_counts < min_count
    rare_count = int(code_counts[rare_mask].sum())
    rare_rate = rare_count / len(cat_codes) if len(cat_codes) > 0 else 0.0

    cat_map = np.full(n_raw_unique, 1, dtype=np.int32)              # 默认 rare
    non_rare = ~rare_mask
    cat_map[non_rare] = np.arange(2, 2 + int(non_rare.sum()), dtype=np.int32)
    vocab_size = 2 + int(non_rare.sum())

    safe_codes = np.where(cat_codes >= 0, cat_codes, 0)
    encoded = np.where(cat_codes >= 0, cat_map[safe_codes], 0).astype(np.int32)
    return encoded, vocab_size, rare_rate


def qc_gate(df: pd.DataFrame) -> dict:
    """QC: 列序、列数、label 二值、特征非负。archive L129-147。"""
    expected_cols = NUM_COLS + CAT_COLS + ["label"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Column order mismatch: {list(df.columns)[:5]}...")
    if df.shape[1] != 40:
        raise ValueError(f"Column count mismatch: {df.shape[1]}")
    label_values = set(df["label"].unique().tolist())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"Label is not binary: {label_values}")
    for col in NUM_COLS + CAT_COLS:
        if int(df[col].min()) < 0:
            raise ValueError(f"Negative feature index in {col}")
    return {
        "column_order_ok": True,
        "column_count_ok": True,
        "label_binary_ok": True,
        "feature_non_negative_ok": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=10,
                    help="Rare threshold for categorical features (archive default 10)")
    args = ap.parse_args()

    raw_dir, processed_dir = resolve_paths("criteo")
    raw_tsv = raw_dir / "train.txt"
    if not raw_tsv.exists():
        for alt in ("dac.txt", "train_50m.txt", "data.txt"):
            p = raw_dir / alt
            if p.exists():
                raw_tsv = p
                print(f"[criteo] using alt file: {raw_tsv}")
                break
        else:
            print(f"[criteo] ERROR: {raw_dir / 'train.txt'} not found. Run `download.py` first.")
            return 1

    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[criteo] reading {raw_tsv}")
    dtypes: dict = {"label": "int8"}
    dtypes.update({f"I{i}": "float64" for i in range(1, 14)})
    dtypes.update({f"C{i}": "category" for i in range(1, 27)})

    df = pd.read_csv(
        raw_tsv,
        sep="\t",
        header=None,
        names=ALL_COLS_INPUT,
        dtype=dtypes,
    )
    if df.shape[1] != 40:
        print(f"[criteo] ERROR: unexpected input columns: {df.shape[1]}")
        return 1
    rows = int(len(df))
    print(f"[criteo] loaded shape={df.shape}, ctr={df['label'].mean():.6f}")

    missing_rate_raw = {col: float(df[col].isna().mean()) for col in ALL_COLS_INPUT}
    feature_rows: list = []
    feature_vocab: dict = {}

    # I1-I13 数值特征
    print(f"[criteo] encoding numeric features (signed_log1p_square_tokens)")
    for col in NUM_COLS:
        encoded, token_vocab = encode_numeric_column_vectorized(df[col])
        df[col] = encoded
        feature_rows.append({
            "feature": col,
            "type": "numeric_bucketed",
            "missing_rate_raw": missing_rate_raw[col],
            "vocab_size_encoded": int(df[col].max()) + 1,
            "bucket_vocab_before_encode": token_vocab,
        })
        feature_vocab[col] = int(df[col].max()) + 1

    # C1-C26 类别特征
    print(f"[criteo] encoding categorical features (min_count={args.min_count} rare-merging)")
    for col in CAT_COLS:
        encoded, vocab_size, rare_rate = encode_categorical_with_rare(df[col], args.min_count)
        df[col] = encoded
        feature_rows.append({
            "feature": col,
            "type": "categorical",
            "missing_rate_raw": missing_rate_raw[col],
            "rare_rate": float(rare_rate),
            "vocab_size_encoded": vocab_size,
        })
        feature_vocab[col] = vocab_size

    df["label"] = df["label"].astype(np.int32)
    df = df[NUM_COLS + CAT_COLS + ["label"]]                # 重排：label 末尾
    qc = qc_gate(df)

    # 输出
    pkl_path = processed_dir / "data.pkl"
    meta_path = processed_dir / "data_meta.json"
    summary_path = processed_dir / "feature_summary.csv"
    df.to_pickle(pkl_path)
    pd.DataFrame(feature_rows).sort_values(by=["type", "feature"]).to_csv(summary_path, index=False)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "config": {
            "input_path": str(raw_tsv),
            "min_count": args.min_count,
            "numeric_bucket_impl": "signed_log1p_square_tokens",
            "column_order": NUM_COLS + CAT_COLS + ["label"],
        },
        "dataset_summary": {
            "rows": rows,
            "cols": 40,
            "ctr": float(df["label"].mean()),
            "label_pos": int(df["label"].sum()),
            "missing_rate_raw": missing_rate_raw,
        },
        "feature_vocab": feature_vocab,
        "total_vocab": int(sum(feature_vocab.values())),
        "field_index": FIELD_INDEX_CRITEO,
        "field_name_at_index": FIELD_NAME_CRITEO,
        "qc": qc,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[criteo] wrote {pkl_path} ({rows:,} rows, 39 sparse features)")
    print(f"[criteo] wrote {meta_path}")
    print(f"[criteo] wrote {summary_path}")
    print(f"[criteo] field_index={FIELD_INDEX_CRITEO} → '{FIELD_NAME_CRITEO}', "
          f"label CTR={meta['dataset_summary']['ctr']:.4f}, "
          f"total_vocab={meta['total_vocab']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
