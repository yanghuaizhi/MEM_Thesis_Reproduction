"""表 4-1: ECE/AUC/LogLoss 全部方法（11 方法 × 3 数据集 mean±std, ddof=1）。

对应论文 Ch4 主结果表。
输出: results/tables/table_4_1.{md,csv}

CLI:
    python -m reproduction.analysis.tables.table_4_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._common import (
    aggregate_by_method_dataset,
    fmt_mean_std,
    render_markdown_table,
    write_csv,
    write_md,
)


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

METHODS_ORDER = [
    "platt", "ir", "hb",
    "umnn", "neucalib", "desc", "sbcr",
    "umc", "umc_wor", "uamcm", "uamcm_wor",
]
DATASETS_ORDER = ["aliccp", "avazu", "criteo"]


def build_table(records: list) -> tuple:
    """返回 (markdown, headers, rows for csv)。"""
    agg_ece = aggregate_by_method_dataset(records, "ece", ddof=1)
    agg_auc = aggregate_by_method_dataset(records, "auc", ddof=1)
    agg_log = aggregate_by_method_dataset(records, "logloss", ddof=1)

    headers = ["Method"]
    for ds in DATASETS_ORDER:
        headers.extend([f"{ds}_ECE×100", f"{ds}_AUC", f"{ds}_LogLoss"])

    md_rows: list = []
    csv_rows: list = []
    for method in METHODS_ORDER:
        md_row = [method]
        csv_row = [method]
        for ds in DATASETS_ORDER:
            ece = agg_ece.get((ds, method))
            auc = agg_auc.get((ds, method))
            log = agg_log.get((ds, method))
            if ece:
                md_row.append(fmt_mean_std(ece["mean"], ece["std"], decimals=2, pct=True))
                csv_row.extend([ece["mean"], ece["std"]])
            else:
                md_row.append("--")
                csv_row.extend(["", ""])
            if auc:
                md_row.append(f"{auc['mean']:.4f}")
                csv_row.append(auc["mean"])
            else:
                md_row.append("--")
                csv_row.append("")
            if log:
                md_row.append(f"{log['mean']:.4f}")
                csv_row.append(log["mean"])
            else:
                md_row.append("--")
                csv_row.append("")
        md_rows.append(md_row)
        csv_rows.append(csv_row)

    md = (
        "# Table 4-1: ECE/AUC/LogLoss for 11 methods × 3 datasets\n\n"
        "Reported: mean±std (ddof=1), N=3 seeds {1024, 2024, 3024}. "
        "ECE×100 for readability. M=100 bins.\n\n"
        "**Note on statistical methods (platt/ir/hb)**: 这三种方法是数学上"
        "100% deterministic 算法（Platt=sklearn LogisticRegression lbfgs、"
        "IR=PAV 算法、HB=digitize+mean），跨 seed 输出完全唯一，因此 std=0 "
        "是算法本质，与 baiyimeng/UMC 原作一致。神经方法（umnn/neucalib/desc/"
        "sbcr/umc/umc_wor/uamcm/uamcm_wor）才报告真 3-seed 方差。\n\n"
    )
    md += render_markdown_table(headers, md_rows)
    md += "\n_Generated from experiments/runs/main (must pass sanity_check first)._\n"

    csv_headers = ["method"]
    for ds in DATASETS_ORDER:
        csv_headers.extend([f"{ds}_ece_mean", f"{ds}_ece_std",
                            f"{ds}_auc_mean", f"{ds}_logloss_mean"])
    return md, csv_headers, csv_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str,
                    default=str(_PROJECT_ROOT / "results" / "tables"))
    args = ap.parse_args()

    # 延迟 import 避免 circular
    sys.path.insert(0, str(_PROJECT_ROOT))
    from reproduction.analysis.diff_with_paper import load_main_metrics

    records = load_main_metrics()
    md, csv_headers, csv_rows = build_table(records)

    out_md = Path(args.out_dir) / "table_4_1.md"
    out_csv = Path(args.out_dir) / "table_4_1.csv"
    write_md(out_md, md)
    write_csv(out_csv, csv_headers, csv_rows)
    print(f"[table_4_1] {out_md}")
    print(f"[table_4_1] {out_csv}  (records loaded: {len(records)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
