"""AliCCP 预处理 — 阿里电商点击-转化数据集。

输入:  data/raw/aliccp/ 含 4 个 CSV:
       common_features_{train,test}.csv  + sample_skeleton_{train,test}.csv
输出:  data/processed/aliccp/data.pkl

参考: UMC/dataset/aliccp_process.ipynb（保留作权威 ground truth）

预处理流程（多步骤 chunked）:
    1. 解析 common_features (user 特征字典)
    2. 解析 sample_skeleton (item 特征 + label)
    3. 按 common_feature_index 做 left join
    4. 保留论文使用的 14 个稀疏特征 + click label
    5. LabelEncoder 编码

注意:
    - 内存压力大（原始数据 15GB），需 chunked 处理
    - 论文使用 14 个稀疏特征（来自 ipynb 的 save_cols 列表）
    - field_index=0 指首列特征 '101'（用户ID，与 train_neu_ali.py L66 一致）

CLI:
    python -m reproduction.data.preprocess.aliccp [--max-chunks N]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ._common import resolve_paths, encode_sparse_features, write_pkl_and_meta


# 论文采用的 14 个稀疏特征（来自 ipynb 的 save_cols）
USER_FEATURE_KEYS: List[str] = ["101", "121", "122", "124", "125", "126", "127", "128", "129"]
ITEM_FEATURE_KEYS: List[str] = ["205", "206", "207", "216", "301"]
ALL_FEATURE_KEYS: List[str] = USER_FEATURE_KEYS + ITEM_FEATURE_KEYS   # 14
# field_index=0 指向首列 '101'（用户ID）。
# 注：train_neu_ali.py L66 的 Config.field_index 默认 2，但 L981 的 trial() 入口
# dict 覆盖为 0；trial 是 UMC 实际执行入口，所以**运行时生效值是 0**。
# 与 reproduction/configs/datasets/aliccp.yaml `field_index: 0` 一致。
FIELD_INDEX_ALICCP = 0
FIELD_NAME_ALICCP = "101"      # u 信号挂钩特征（用户ID）


# chunk sizes（与 ipynb 一致）
COMMON_FEATURES_CHUNK = 100_000
SAMPLE_SKELETON_CHUNK = 2_500_000


def feature_list_split(x: str) -> dict:
    """解析 feature_list 字段（'\\x01' 分隔多值）。"""
    d: defaultdict = defaultdict(list)
    if not isinstance(x, str):
        return dict(d)
    for fea in x.split("\x01"):
        parts = re.split("\x02", fea)
        if len(parts) >= 2:
            d[parts[0]].append(parts[1])
    return dict(d)


def parse_user_features(chunk: pd.DataFrame) -> pd.DataFrame:
    """从 common_features chunk 解析 user 字段（USER_FEATURE_KEYS）。"""
    chunk["fea_dict"] = chunk["feature_list2"].map(feature_list_split)
    for key in USER_FEATURE_KEYS:
        chunk[key] = chunk["fea_dict"].map(
            lambda d: int(d[key][0].split("\x03")[0]) if key in d and d[key] else 0
        )
    chunk = chunk.drop(columns=["feature_num2", "feature_list2", "fea_dict"])
    return chunk


def parse_item_features(chunk: pd.DataFrame) -> pd.DataFrame:
    """从 sample_skeleton chunk 解析 item 字段（ITEM_FEATURE_KEYS + click）。"""
    chunk["fea_dict"] = chunk["feature_list1"].map(feature_list_split)
    for key in ITEM_FEATURE_KEYS:
        chunk[key] = chunk["fea_dict"].map(
            lambda d: int(d[key][0].split("\x03")[0]) if key in d and d[key] else 0
        )
    chunk = chunk.drop(columns=["feature_num1", "feature_list1", "fea_dict"])
    return chunk


def process_split(
    raw_dir: Path,
    split: str,                        # "train" or "test"
    max_chunks: int = None,
) -> pd.DataFrame:
    """处理 train 或 test 一个 split，返回合并后的 DataFrame。"""
    common_csv = raw_dir / f"common_features_{split}.csv"
    sample_csv = raw_dir / f"sample_skeleton_{split}.csv"
    if not common_csv.exists() or not sample_csv.exists():
        raise FileNotFoundError(f"missing {common_csv} or {sample_csv}")

    # 1. 解析 common_features（user）
    print(f"[aliccp/{split}] parsing common_features...")
    user_chunks = []
    fea_iter = pd.read_csv(
        common_csv,
        header=None,
        names=["common_feature_index", "feature_num2", "feature_list2"],
        iterator=True,
        chunksize=COMMON_FEATURES_CHUNK,
    )
    for i, chunk in enumerate(fea_iter):
        if max_chunks is not None and i >= max_chunks:
            break
        user_chunks.append(parse_user_features(chunk))
        if (i + 1) % 5 == 0:
            print(f"  user chunk {i + 1}: shape={chunk.shape}")
    user_df = pd.concat(user_chunks, ignore_index=True)
    print(f"[aliccp/{split}] user_df shape: {user_df.shape}")

    # 2. 解析 sample_skeleton（item + click）
    print(f"[aliccp/{split}] parsing sample_skeleton...")
    item_chunks = []
    sam_iter = pd.read_csv(
        sample_csv,
        header=None,
        names=["sample_id", "click", "conversion", "common_feature_index",
               "feature_num1", "feature_list1"],
        iterator=True,
        chunksize=SAMPLE_SKELETON_CHUNK,
    )
    for i, chunk in enumerate(sam_iter):
        if max_chunks is not None and i >= max_chunks:
            break
        item_chunks.append(parse_item_features(chunk))
        print(f"  item chunk {i + 1}: shape={chunk.shape}")
    item_df = pd.concat(item_chunks, ignore_index=True)
    print(f"[aliccp/{split}] item_df shape: {item_df.shape}")

    # 3. left join user → item
    keep_user = USER_FEATURE_KEYS + ["common_feature_index"]
    keep_item = ITEM_FEATURE_KEYS + ["common_feature_index", "sample_id", "click"]
    merged = item_df[keep_item].merge(user_df[keep_user], how="left",
                                     on="common_feature_index")
    print(f"[aliccp/{split}] merged shape: {merged.shape}")
    return merged[ALL_FEATURE_KEYS + ["click"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="每个 split 处理多少个 chunk（调试用，默认全量）")
    args = ap.parse_args()

    raw_dir, processed_dir = resolve_paths("aliccp")
    print(f"[aliccp] raw={raw_dir}, processed={processed_dir}")

    expected = [
        "common_features_train.csv", "common_features_test.csv",
        "sample_skeleton_train.csv", "sample_skeleton_test.csv",
    ]
    missing = [f for f in expected if not (raw_dir / f).exists()]
    if missing:
        print(f"[aliccp] ERROR: missing files in {raw_dir}: {missing}")
        print(f"[aliccp] Run `python -m reproduction.data.download --dataset aliccp` for download guide.")
        return 1

    train = process_split(raw_dir, "train", max_chunks=args.max_chunks)
    test = process_split(raw_dir, "test", max_chunks=args.max_chunks)

    # 合并 train + test（train_neu_ali.py 内做 60/20/20 split）
    full = pd.concat([train, test], ignore_index=True)
    print(f"[aliccp] combined shape: {full.shape}")

    encoded, vocab = encode_sparse_features(
        full,
        sparse_features=ALL_FEATURE_KEYS,
        label_col="click",
        encoder="ordinal",
    )
    write_pkl_and_meta(
        encoded,
        processed_dir,
        sparse_features=ALL_FEATURE_KEYS,
        vocab_sizes=vocab,
        field_index=FIELD_INDEX_ALICCP,
        label_col="click",
    )
    print(f"[aliccp] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
