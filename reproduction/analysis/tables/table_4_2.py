"""表 4-2: UAMCM vs UMC 改善百分比 + seed 一致性。

对应论文 Ch4 §4.x。
输出: results/tables/table_4_2.{md,csv}

CLI:
    python -m reproduction.analysis.tables.table_4_2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._common import (
    aggregate_by_method_dataset,
    render_markdown_table,
    write_csv,
    write_md,
)


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent


def build_table(records: list) -> tuple:
    agg = aggregate_by_method_dataset(records, "ece", ddof=1)

    headers = ["Dataset", "UMC ECE×100", "UAMCM ECE×100",
               "Improvement %", "Seed Consistency (UAMCM<UMC)"]
    md_rows: list = []
    csv_rows: list = []
    for ds in ("aliccp", "avazu", "criteo"):
        umc = agg.get((ds, "umc"))
        uamcm = agg.get((ds, "uamcm"))
        if not umc or not uamcm:
            md_rows.append([ds, "--", "--", "--", "--"])
            csv_rows.append([ds, "", "", "", ""])
            continue
        imp_pct = 100 * (umc["mean"] - uamcm["mean"]) / umc["mean"] if umc["mean"] else 0
        # seed consistency
        per_seed: dict = {1024: {}, 2024: {}, 3024: {}}
        for r in records:
            if r["dataset"] != ds or "ece" not in r:
                continue
            if r["method"] in ("umc", "uamcm"):
                per_seed[r["seed"]][r["method"]] = r["ece"]
        wins = sum(
            1 for s in per_seed.values()
            if "umc" in s and "uamcm" in s and s["uamcm"] < s["umc"]
        )
        md_rows.append([
            ds,
            f"{umc['mean']*100:.2f}±{umc['std']*100:.2f}",
            f"{uamcm['mean']*100:.2f}±{uamcm['std']*100:.2f}",
            f"{imp_pct:.1f}%",
            f"{wins}/3",
        ])
        csv_rows.append([ds, umc["mean"], umc["std"], uamcm["mean"], uamcm["std"],
                         imp_pct, wins])

    md = (
        "# Table 4-2: UAMCM vs UMC — improvement & seed consistency\n\n"
        "Improvement % = (UMC_ECE - UAMCM_ECE) / UMC_ECE × 100.\n"
        "Seed consistency: 多少个 seed 上 UAMCM ECE < UMC ECE (期望 ≥ 1/3 通过谨慎推荐阈值).\n"
        "ECE×100 with std (ddof=1).\n\n"
    )
    md += render_markdown_table(headers, md_rows)
    md += "\n_Use ECE表述: 降低 X% (plan §B 第 10 条)._\n"
    csv_headers = ["dataset", "umc_ece_mean", "umc_ece_std",
                   "uamcm_ece_mean", "uamcm_ece_std", "improvement_pct", "seed_wins"]
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
    write_md(Path(args.out_dir) / "table_4_2.md", md)
    write_csv(Path(args.out_dir) / "table_4_2.csv", h, rows)
    print(f"[table_4_2] written (records={len(records)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
