"""Criteo 预处理 — chunked 两遍扫，避免一次性加载 11GB TSV。

挂死现场（2026-05-14 14:12）：pd.read_csv 一次读 11G train.txt，在 vast 网络挂载存储 +
pandas C engine 单线程解析下长时间无 stdout flush，看起来"挂死"。

工程修改（**不影响数值，与 archive ground truth 算法 1:1 等价**）：
- 流式 chunked 读，单 chunk ~640MB DataFrame 峰值
- Pass 1: 累积 numeric (sign, bucket) unique 集 + 全局 max_bucket + cat value_counts
- Pass 2: 流式编码，写入 pre-allocated int32 ndarray (N, 39)

算法等价性证明：
- Numeric: sklearn LabelEncoder.fit_transform 顺序 = sorted(unique compound) 分配 0,1,2,...
  Chunked 累积所有 (sign, bucket) → 重建 compound = sign * (global_max_bucket+1) + bucket
  → sorted 分配 id ⇒ 与 archive 完全等价
- Categorical: pandas Categorical.cat.codes 顺序 = sorted(categories) lexicographic
  Chunked 累积 value_counts → sorted(non_rare) 分配 id 2,3,4,... ⇒ 与 archive 完全等价

ground truth: /Users/y/Research_MEM/10_research_archive/dataset/criteo/preprocess_criteo.py

CLI:
    python -m reproduction.data.preprocess.criteo [--min-count 10] [--chunk-rows 2000000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ._common import resolve_paths


NUM_COLS: List[str] = [f"I{i}" for i in range(1, 14)]
CAT_COLS: List[str] = [f"C{i}" for i in range(1, 27)]
ALL_FEATURES: List[str] = NUM_COLS + CAT_COLS
ALL_COLS_INPUT: List[str] = ["label"] + NUM_COLS + CAT_COLS
FIELD_INDEX_CRITEO = 23
FIELD_NAME_CRITEO = "C11"
DEFAULT_CHUNK_ROWS = 2_000_000               # ~52MB raw TSV / chunk，单 chunk DF 峰值 ~640MB


def _read_chunks(path: Path, chunk_rows: int):
    """Stream TSV in row chunks. CAT cols as nullable string (NOT category)."""
    dtypes: dict = {"label": "int8"}
    dtypes.update({f"I{i}": "float64" for i in range(1, 14)})
    dtypes.update({f"C{i}": "string" for i in range(1, 27)})
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=ALL_COLS_INPUT,
        dtype=dtypes,
        chunksize=chunk_rows,
        engine="c",
    )


def _numeric_sign_bucket(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (sign, bucket, is_missing). 1:1 等价 archive encode_numeric_column_vectorized 的前半段。"""
    is_missing = np.isnan(values)
    safe = np.where(is_missing, 0.0, values)
    bucket = np.floor(np.log1p(np.abs(safe)) ** 2).astype(np.int64)
    sign = np.where(
        is_missing, 0,
        np.where(safe < 0, 2, np.where(safe > 0, 3, 1)),
    ).astype(np.int64)
    return sign, bucket, is_missing


def _pass1_scan_vocab(path: Path, chunk_rows: int):
    """Pass 1: stream-scan to collect vocab info per column."""
    num_pairs: Dict[str, set] = {c: set() for c in NUM_COLS}
    num_max_bucket: Dict[str, int] = {c: 0 for c in NUM_COLS}
    cat_counts: Dict[str, Counter] = {c: Counter() for c in CAT_COLS}
    col_missing: Dict[str, int] = {c: 0 for c in ALL_COLS_INPUT}
    total_rows = 0
    label_pos = 0

    for chunk_idx, chunk in enumerate(_read_chunks(path, chunk_rows)):
        n = len(chunk)
        total_rows += n
        label_pos += int(chunk["label"].sum())

        for col in NUM_COLS:
            vals = chunk[col].values.astype(np.float64)
            sign, bucket, is_missing = _numeric_sign_bucket(vals)
            col_missing[col] += int(is_missing.sum())
            if bucket.size:
                mb = int(bucket.max())
                if mb > num_max_bucket[col]:
                    num_max_bucket[col] = mb
            # Pack (sign, bucket) into int64 for fast numpy unique + set update
            packed = (sign << np.int64(40)) | bucket
            num_pairs[col].update(np.unique(packed).tolist())

        for col in CAT_COLS:
            s = chunk[col]
            col_missing[col] += int(s.isna().sum())
            valid = s.dropna()
            if len(valid) > 0:
                vc = valid.value_counts(sort=False)
                for k, v in vc.items():
                    cat_counts[col][k] += int(v)

        if chunk_idx == 0 or (chunk_idx + 1) % 5 == 0:
            print(f"[criteo]   Pass 1 chunk {chunk_idx+1} rows_so_far={total_rows:,}", flush=True)

    return num_pairs, num_max_bucket, cat_counts, col_missing, total_rows, label_pos


def _build_encoders(num_pairs, num_max_bucket, cat_counts, total_rows, min_count):
    """Pass 1.5: build vocab → id encoders matching archive semantics."""
    num_encoders: Dict[str, Tuple[int, np.ndarray]] = {}
    cat_encoders: Dict[str, Tuple[Dict[str, int], int, float]] = {}
    feature_extra: Dict[str, dict] = {}

    for col in NUM_COLS:
        max_bucket_p1 = num_max_bucket[col] + 1            # +1 等价 archive `int(bucket.max()) + 1`
        compounds: List[int] = []
        for packed in num_pairs[col]:
            s_val = packed >> 40
            b_val = packed & ((1 << 40) - 1)
            compounds.append(int(s_val * max_bucket_p1 + b_val))
        compound_sorted = sorted(compounds)
        # Flat lookup: index = compound value, value = id; sign max=3, so 4*max_bucket_p1 suffices
        lookup_size = 4 * max_bucket_p1
        lookup = np.full(lookup_size, -1, dtype=np.int32)
        for i, c_val in enumerate(compound_sorted):
            lookup[c_val] = i
        num_encoders[col] = (max_bucket_p1, lookup)
        feature_extra[col] = {"type": "numeric_bucketed", "vocab": len(compound_sorted)}

    for col in CAT_COLS:
        counts = cat_counts[col]
        all_sorted = sorted(counts.keys())                  # lexicographic = archive cat.categories 顺序
        non_rare = [v for v in all_sorted if counts[v] >= min_count]
        rare_count = sum(c for v, c in counts.items() if c < min_count)
        rare_rate = rare_count / total_rows if total_rows > 0 else 0.0
        val_to_id = {v: i + 2 for i, v in enumerate(non_rare)}      # 0=missing, 1=rare, 2+=normal
        vocab_size = 2 + len(non_rare)
        cat_encoders[col] = (val_to_id, vocab_size, rare_rate)
        feature_extra[col] = {
            "type": "categorical",
            "raw_unique": len(all_sorted),
            "rare_rate": rare_rate,
            "vocab": vocab_size,
        }

    return num_encoders, cat_encoders, feature_extra


def _pass2_encode(path, total_rows, chunk_rows, num_encoders, cat_encoders):
    """Pass 2: stream-apply encoders, write into pre-allocated int32 arrays."""
    encoded = np.empty((total_rows, 39), dtype=np.int32)
    labels = np.empty(total_rows, dtype=np.int32)
    row_offset = 0

    for chunk_idx, chunk in enumerate(_read_chunks(path, chunk_rows)):
        n = len(chunk)
        end = row_offset + n
        labels[row_offset:end] = chunk["label"].values.astype(np.int32)

        for ci, col in enumerate(NUM_COLS):
            max_bucket_p1, lookup = num_encoders[col]
            vals = chunk[col].values.astype(np.float64)
            sign, bucket, _ = _numeric_sign_bucket(vals)
            compound = (sign * max_bucket_p1 + bucket).astype(np.int64)
            compound = np.clip(compound, 0, len(lookup) - 1)        # defensive; Pass 1 already saw all
            encoded[row_offset:end, ci] = lookup[compound]

        for ci, col in enumerate(CAT_COLS):
            val_to_id, _, _ = cat_encoders[col]
            s = chunk[col]
            not_na = s.notna().values
            s_filled = s.fillna("")
            mapped = s_filled.map(val_to_id).fillna(1).astype(np.int32).values
            codes = np.where(not_na, mapped, 0).astype(np.int32)    # 0 if NA, else mapped (rare=1)
            encoded[row_offset:end, 13 + ci] = codes

        row_offset = end
        if chunk_idx == 0 or (chunk_idx + 1) % 5 == 0:
            print(f"[criteo]   Pass 2 chunk {chunk_idx+1} rows_encoded={row_offset:,}", flush=True)

    assert row_offset == total_rows, f"row count mismatch: {row_offset} vs {total_rows}"
    return encoded, labels


def qc_gate(df: pd.DataFrame) -> dict:
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
    ap.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                    help=f"Rows per streaming chunk (default {DEFAULT_CHUNK_ROWS:,})")
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
    input_stat = os.stat(raw_tsv)
    print(f"[criteo] streaming chunked preprocess (chunk_rows={args.chunk_rows:,})", flush=True)
    print(f"[criteo] input: {raw_tsv} ({input_stat.st_size:,} bytes)", flush=True)

    print(f"[criteo] Pass 1/2: scanning for vocab", flush=True)
    num_pairs, num_max_bucket, cat_counts, col_missing, total_rows, label_pos = \
        _pass1_scan_vocab(raw_tsv, args.chunk_rows)
    print(f"[criteo] Pass 1 done: rows={total_rows:,} ctr={label_pos/total_rows:.6f}", flush=True)

    num_encoders, cat_encoders, feature_extra = \
        _build_encoders(num_pairs, num_max_bucket, cat_counts, total_rows, args.min_count)
    del num_pairs, cat_counts

    print(f"[criteo] Pass 2/2: encoding", flush=True)
    encoded, labels = _pass2_encode(raw_tsv, total_rows, args.chunk_rows, num_encoders, cat_encoders)
    del num_encoders, cat_encoders

    df = pd.DataFrame(encoded, columns=NUM_COLS + CAT_COLS)
    df["label"] = labels
    del encoded, labels

    qc = qc_gate(df)
    print(f"[criteo] QC passed, shape={df.shape}, ctr={df['label'].mean():.6f}", flush=True)

    missing_rate_raw = {col: col_missing[col] / total_rows for col in ALL_COLS_INPUT}
    feature_rows = []
    for col in NUM_COLS:
        feature_rows.append({
            "feature": col,
            "type": "numeric_bucketed",
            "missing_rate_raw": missing_rate_raw[col],
            "raw_unique_non_missing": int(df[col].nunique()),
            "rare_rate": 0.0,
            "vocab_size_encoded": int(df[col].max()) + 1,
            "bucket_vocab_before_encode": feature_extra[col]["vocab"],
        })
    for col in CAT_COLS:
        feature_rows.append({
            "feature": col,
            "type": "categorical",
            "missing_rate_raw": missing_rate_raw[col],
            "raw_unique_non_missing": feature_extra[col]["raw_unique"],
            "rare_rate": float(feature_extra[col]["rare_rate"]),
            "vocab_size_encoded": feature_extra[col]["vocab"],
            "bucket_vocab_before_encode": None,
        })

    feature_vocab = {col: int(df[col].max()) + 1 for col in NUM_COLS + CAT_COLS}
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
            "chunk_rows": args.chunk_rows,
            "numeric_bucket_impl": "signed_log1p_square_tokens",
            "column_order": NUM_COLS + CAT_COLS + ["label"],
            "implementation": "chunked_two_pass",
        },
        "input_file": {
            "size_bytes": int(input_stat.st_size),
            "mtime": datetime.fromtimestamp(input_stat.st_mtime).isoformat(timespec="seconds"),
        },
        "dataset_summary": {
            "rows": int(total_rows),
            "cols": 40,
            "ctr": float(df["label"].mean()),
            "label_pos": int(df["label"].sum()),
            "label_neg": int(total_rows - int(df["label"].sum())),
            "missing_rate_raw": missing_rate_raw,
        },
        "feature_vocab": feature_vocab,
        "total_vocab": int(sum(feature_vocab.values())),
        "field_index": FIELD_INDEX_CRITEO,
        "field_name_at_index": FIELD_NAME_CRITEO,
        "qc": qc,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[criteo] wrote {pkl_path} ({total_rows:,} rows, 39 sparse features)")
    print(f"[criteo] wrote {meta_path}")
    print(f"[criteo] wrote {summary_path}")
    print(f"[criteo] field_index={FIELD_INDEX_CRITEO} → '{FIELD_NAME_CRITEO}', "
          f"label CTR={meta['dataset_summary']['ctr']:.4f}, "
          f"total_vocab={meta['total_vocab']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
