"""表 4-4: UMC 系列消融（UMC vs UMC-WOR vs UAMCM vs UAMCM-WOR）。

对应论文 Ch4 ranking loss 消融分析。
输出: results/tables/table_4_4.{md,csv}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._common import aggregate_by_method_dataset, render_markdown_table, write_csv, write_md


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

ABLATION_METHODS = ["umc_wor", "umc", "uamcm_wor", "uamcm"]


def build_table(records: list) -> tuple:
    agg = aggregate_by_method_dataset(records, "ece", ddof=1)

    headers = ["Method"] + ["aliccp ECE×100", "avazu ECE×100", "criteo ECE×100"]
    md_rows: list = []
    csv_rows: list = []
    for method in ABLATION_METHODS:
        md_row = [method]
        csv_row = [method]
        for ds in ("aliccp", "avazu", "criteo"):
            v = agg.get((ds, method))
            if v:
                md_row.append(f"{v['mean']*100:.2f}±{v['std']*100:.2f}")
                csv_row.extend([v["mean"], v["std"]])
            else:
                md_row.append("--")
                csv_row.extend(["", ""])
        md_rows.append(md_row)
        csv_rows.append(csv_row)

    md = (
        "# Table 4-4: Ranking loss ablation — UMC vs UMC-WOR vs UAMCM vs UAMCM-WOR\n\n"
        "Each row: ECE×100, mean±std (ddof=1, N=3 seeds, M=100 bins).\n"
        "对比目的: 验证 rescaling=True (ranking loss) 是否带来稳定改善。\n\n"
    )
    md += render_markdown_table(headers, md_rows)
    csv_headers = ["method"] + [f"{ds}_ece_{stat}" for ds in ("aliccp", "avazu", "criteo")
                                for stat in ("mean", "std")]
    return md, csv_headers, csv_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str,
                    default=str(_PROJECT_ROOT / "results" / "tables"))
    args = ap.parse_args()
    sys.path.insert(0, str(_PROJECT_ROOT))
    from reproduction.analysis.diff_with_paper import load_main_metrics

    records = load_main_metrics()
    md, h, rows = build_table(records)
    write_md(Path(args.out_dir) / "table_4_4.md", md)
    write_csv(Path(args.out_dir) / "table_4_4.csv", h, rows)
    print(f"[table_4_4] written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
