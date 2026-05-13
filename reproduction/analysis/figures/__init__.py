"""figures 子包 — 各章节图表生成器（PDF + PNG）。

每个 *.py 都可独立 CLI 运行，输出 results/figures/<name>.{pdf,png}。

文件 → 论文图的映射:
    fig_3_pcoc_u_dist.py    Ch3 图 3-1~3-4: PCOC-u 分布
    fig_3_heatmap.py        Ch3 图 3-2: AliCCP E[Y|p,u] 热力图
    fig_4_main.py           Ch4 图 4-1: 方法对比柱状图
    fig_4_2_shuffled.py     Ch4 图 4-2: shuffled-u 消融对比

无数据时优雅退化（写占位 PDF 含 "data missing" 说明）。
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_matplotlib():
    """惰性 import matplotlib（避免无 GUI 环境下 import 崩）。"""
    import matplotlib
    matplotlib.use("Agg")              # headless backend
    import matplotlib.pyplot as plt
    return matplotlib, plt


def save_placeholder(out_path: Path, message: str) -> None:
    """生成占位 PDF（数据未生成时使用）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mpl, plt = get_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            wrap=True, fontsize=12, color="gray")
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
