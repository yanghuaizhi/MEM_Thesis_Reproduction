"""结果聚合表的共享工具。

设计:
    - 所有 tables/*.py 通过 diff_with_paper.load_main_metrics 等读数据
    - markdown 表格生成走本模块的 render_markdown_table
    - 每个 tables/X.py 输出 results/tables/X.md + X.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def fmt_mean_std(mean: float, std: float, decimals: int = 2, pct: bool = False) -> str:
    """格式化 mean±std 字符串（论文风格）。"""
    mul = 100 if pct else 1
    return f"{mean*mul:.{decimals}f}±{std*mul:.{decimals}f}"


def std_ddof1(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)


def render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """生成对齐的 markdown 表。"""
    out: List[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def write_csv(path: Path, headers: List[str], rows: List[List[Any]]) -> None:
    """写 CSV（聚合数据，便于 Excel / pandas 后续处理）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def aggregate_by_method_dataset(
    records: List[Dict[str, Any]],
    metric: str = "ece",
    ddof: int = 1,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """按 (dataset, method) 分组聚合，返回 {key: {mean, std, n}}。"""
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for r in records:
        if metric not in r:
            continue
        grouped.setdefault((r["dataset"], r["method"]), []).append(float(r[metric]))
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for k, vals in grouped.items():
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        out[k] = {"mean": mean, "std": std_ddof1(vals), "n": n}
    return out
