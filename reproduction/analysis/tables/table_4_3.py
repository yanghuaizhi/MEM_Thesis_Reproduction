"""表 4-3: 统计方法 vs 神经方法对比（Criteo 排名等）。

对应论文 Ch4 §4.x + S1 决策依据。
输出: results/tables/table_4_3.{md,csv}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._common import aggregate_by_method_dataset, render_markdown_table, write_csv, write_md


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

STATISTICAL = ["platt", "ir", "hb"]
NEURAL = ["umnn", "neucalib", "desc", "sbcr", "umc", "umc_wor", "uamcm", "uamcm_wor"]


def build_table(records: list) -> tuple:
    agg = aggregate_by_method_dataset(records, "ece", ddof=1)

    headers = ["Dataset", "Best Statistical (ECE×100)", "Best Neural (ECE×100)",
               "Top-3 by ECE", "S1 Eligible (stat in top-3)"]
    md_rows: list = []
    csv_rows: list = []
    for ds in ("aliccp", "avazu", "criteo"):
        ds_methods = {m: v for (d, m), v in agg.items() if d == ds}
        if not ds_methods:
            md_rows.append([ds, "--", "--", "--", "--"])
            csv_rows.append([ds, "", "", "", ""])
            continue
        sorted_methods = sorted(ds_methods.items(), key=lambda kv: kv[1]["mean"])
        top3 = [m for m, _ in sorted_methods[:3]]
        s1_eligible = any(m in STATISTICAL for m in top3)
        best_stat = next(((m, v) for m, v in sorted_methods if m in STATISTICAL), None)
        best_neur = next(((m, v) for m, v in sorted_methods if m in NEURAL), None)
        md_rows.append([
            ds,
            f"{best_stat[0]}: {best_stat[1]['mean']*100:.2f}" if best_stat else "--",
            f"{best_neur[0]}: {best_neur[1]['mean']*100:.2f}" if best_neur else "--",
            ", ".join(top3),
            "Yes" if s1_eligible else "No",
        ])
        csv_rows.append([
            ds,
            best_stat[0] if best_stat else "",
            best_stat[1]["mean"] if best_stat else "",
            best_neur[0] if best_neur else "",
            best_neur[1]["mean"] if best_neur else "",
            ";".join(top3),
            s1_eligible,
        ])

    md = (
        "# Table 4-3: Statistical vs Neural method comparison\n\n"
        "S1 决策（plan §A.4.2）: Criteo 上 IR/Platt/HB 排进 top-3 = S1 支持。\n\n"
    )
    md += render_markdown_table(headers, md_rows)
    csv_headers = ["dataset", "best_stat_method", "best_stat_ece",
                   "best_neural_method", "best_neural_ece", "top3", "s1_eligible"]
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
    write_md(Path(args.out_dir) / "table_4_3.md", md)
    write_csv(Path(args.out_dir) / "table_4_3.csv", h, rows)
    print(f"[table_4_3] written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
