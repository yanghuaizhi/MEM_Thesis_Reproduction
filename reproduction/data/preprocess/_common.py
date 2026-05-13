"""共享预处理工具：LabelEncoder + parquet 输出 + 路径解析。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent        # 30_reproduction/


def resolve_paths(dataset: str) -> Tuple[Path, Path]:
    """返回 (raw_dir, processed_dir)，按 _paths.py 解析。"""
    sys.path.insert(0, str(_PROJECT_ROOT / "UMC"))
    from _paths import DATA_ROOT                  # type: ignore

    processed_dir = Path(DATA_ROOT) / dataset
    raw_dir = Path(DATA_ROOT).parent / "raw" / dataset
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir


def encode_sparse_features(
    df,
    sparse_features: Iterable[str],
    label_col: str = "click",
    encoder: str = "ordinal",
):
    """对稀疏特征列做 LabelEncoder/OrdinalEncoder，返回新 DataFrame + 特征字典。

    Args:
        df: pandas DataFrame
        sparse_features: 稀疏特征列名
        label_col: 标签列名
        encoder: "ordinal" (sklearn.OrdinalEncoder) or "label"
                 (LabelEncoder 各列独立)

    Returns:
        (encoded_df, vocab_sizes) — encoded_df 列序 = sparse_features + [label_col]
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    sparse_features = list(sparse_features)
    df = df.copy()
    vocab_sizes = {}

    if encoder == "ordinal":
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[sparse_features] = oe.fit_transform(df[sparse_features].astype(str))
        for col, cats in zip(sparse_features, oe.categories_):
            vocab_sizes[col] = len(cats)
    elif encoder == "label":
        for col in sparse_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            vocab_sizes[col] = len(le.classes_)
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    return df[sparse_features + [label_col]], vocab_sizes


def write_pkl_and_meta(
    df,
    processed_dir: Path,
    sparse_features: List[str],
    vocab_sizes: dict,
    field_index: int,
    label_col: str = "click",
) -> None:
    """输出 data.pkl + feature_meta.json 到 processed_dir。

    与 UMC/train_neu_*.py get_data() 期望的 schema 对齐：
        - data.pkl 是 pandas DataFrame
        - 列顺序: <稀疏特征...> + [label_col]
        - get_data() 取所有列除最后一列作 feature_names，最后一列作 label
    """
    import pandas as pd

    pkl_path = processed_dir / "data.pkl"
    df.to_pickle(pkl_path)

    meta = {
        "dataset": processed_dir.name,
        "sparse_features": sparse_features,
        "label_col": label_col,
        "field_index": field_index,           # u 信号字段位置（plan §B 第 9 条）
        "vocab_sizes": vocab_sizes,
        "n_samples": int(len(df)),
        "ctr": float(df[label_col].mean()),
    }
    meta_path = processed_dir / "feature_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))

    print(f"[preprocess] wrote {pkl_path} ({len(df):,} rows, {len(sparse_features)} sparse features)")
    print(f"[preprocess] wrote {meta_path}")
    print(f"[preprocess] field_index={field_index}  label CTR={meta['ctr']:.4f}")
