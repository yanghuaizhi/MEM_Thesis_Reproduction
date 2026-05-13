"""全局 seed 控制，保证训练可复现。

UMC/{pretrain,train_*}.py 已有自己的 setup_seed；本模块为 reproduction 层
（orchestrator、analysis、data 预处理）提供等价实现，参数与 UMC/ 完全一致。

复现的 seed 集（plan §C.2）:
    pretrain seed = 1024（固定）
    calib seeds   = {1024, 2024, 3024}
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any


def setup_seed(seed: int) -> None:
    """对 torch / numpy / random / PYTHONHASHSEED 设置 seed，并强制 cudnn 确定性。

    与 UMC/pretrain.py L31-42 逻辑一致；保证三处入口的种子语义相同。
    cudnn.benchmark=False 是 plan §B 第 7/8 条避坑要求（Tier 3 红线）。
    """
    import numpy as np
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def derive_seed(base_seed: int, *salts: Any) -> int:
    """从 base_seed + 任意 salts 派生子 seed（确定性 hash）。

    用例:
        - 派生 DataLoader 的 generator seed
        - 派生 shuffled-u 消融时的打乱 seed（保证可复现）
        - 派生不同方法/epoch 的 seed 切片
    """
    key = f"{base_seed}:" + ":".join(str(s) for s in salts)
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


def torch_generator(seed: int):
    """构造一个固定 seed 的 torch.Generator（用于 DataLoader generator=）。"""
    import torch

    return torch.Generator().manual_seed(int(seed))


if __name__ == "__main__":
    # 自检：派生应当确定性，调用两次结果一致
    s1 = derive_seed(1024, "aliccp", "umc")
    s2 = derive_seed(1024, "aliccp", "umc")
    assert s1 == s2, "derive_seed not deterministic"
    s3 = derive_seed(1024, "aliccp", "uamcm")
    assert s1 != s3, "derive_seed not discriminative"
    print(f"derive_seed(1024, 'aliccp', 'umc')   = {s1}")
    print(f"derive_seed(1024, 'aliccp', 'uamcm') = {s3}")
    print("seed.py self-check OK")
