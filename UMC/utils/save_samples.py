"""
save_samples.py — sample-level 数据保存工具
============================================
用法: 在 train_neu_*.py 的 loss-best / ece-best 评估完成后调用.
不修改训练脚本的核心逻辑, 只在评估阶段附加保存.
"""

import os
import numpy as np


def save_sample_level(
    test_y,
    test_y_pred,
    test_y_pred_calib,
    test_sigma2,
    save_path,
    test_alpha=None,
):
    """保存 sample-level 四元组到 npz 文件.

    Args:
        test_y: (N,) ground truth labels
        test_y_pred: (N,) uncalibrated predictions (sigmoid of backbone logit)
        test_y_pred_calib: (N,) calibrated predictions
        test_sigma2: (N,) raw PE variance from backbone
        save_path: 输出路径 (建议 .npz 后缀)
        test_alpha: (N,) or (N,K) optional alpha/router weights
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    data = {
        "y_true": np.asarray(test_y, dtype=np.float32),
        "y_pred_uncalib": np.asarray(test_y_pred, dtype=np.float32),
        "y_pred_calib": np.asarray(test_y_pred_calib, dtype=np.float32),
        "sigma2": np.asarray(test_sigma2, dtype=np.float32),
    }

    if test_alpha is not None:
        alpha = np.asarray(test_alpha, dtype=np.float32)
        if alpha.ndim == 1:
            data["alpha"] = alpha
        else:
            for k in range(alpha.shape[1]):
                data[f"w_{k}"] = alpha[:, k]

    np.savez_compressed(save_path, **data)
    n = len(data["y_true"])
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"sample_level_saved path={save_path} samples={n} size={size_mb:.1f}MB")
