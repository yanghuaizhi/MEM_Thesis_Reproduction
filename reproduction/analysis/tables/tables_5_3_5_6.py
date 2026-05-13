"""Ch5 表 5-3 ~ 5-6: 三重门槛 + shuffled-u u_mode 消融。

对应论文 Ch5：
    Table 5-3: 数据集模式 + 诊断预判
    Table 5-4: 三重门槛通过率（UMC CV%, UAMCM CV%, shuffled-u 验证）
    Table 5-5: u_mode 三态对比（PE / shuffled / logit）
    Table 5-6: P5 论证支点 — 诊断预判 vs 实验方向命中率

输出: results/tables/table_5_{3,4,5,6}.{md,csv}
"""

from __future__ import annotations

import argparse
import math
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


def cv_pct(mean: float, std: float) -> float:
    return 100 * std / abs(mean) if mean else 0.0


def build_table_5_3() -> str:
    """Table 5-3: 数据集模式 + 诊断预判（论文 Ch3 文字结论）。"""
    sys.path.insert(0, str(_PROJECT_ROOT))
    from reproduction.analysis.diff_with_paper import PAPER_REFERENCE
    headers = ["Dataset", "全局 PCOC", "过预测桶/20", "PCOC CV%", "模式", "u 信号预判"]
    rows = [
        ["aliccp", f"{PAPER_REFERENCE['aliccp']['pcoc']:.3f}",
         str(PAPER_REFERENCE["aliccp"]["over_predict_bins_out_of_20"]),
         f"{PAPER_REFERENCE['aliccp']['pcoc_cv_pct']:.2f}",
         "A 强过预测", "全域有效（shuffled 应显著恶化）"],
        ["avazu", f"{PAPER_REFERENCE['avazu']['pcoc']:.3f}",
         str(PAPER_REFERENCE["avazu"]["over_predict_bins_out_of_20"]),
         f"{PAPER_REFERENCE['avazu']['pcoc_cv_pct']:.2f}",
         "C 非单调混合", "方向混乱（shuffled 应不显著恶化）"],
        ["criteo", f"{PAPER_REFERENCE['criteo']['pcoc']:.3f}",
         str(PAPER_REFERENCE["criteo"]["over_predict_bins_out_of_20"]),
         f"{PAPER_REFERENCE['criteo']['pcoc_cv_pct']:.2f}",
         "B 弱欠预测", "局部显著（shuffled 应显著恶化）"],
    ]
    md = "# Table 5-3: 数据集模式与诊断预判（基于论文 v1.13 参考；复现后用 L1 报告替换）\n\n"
    md += render_markdown_table(headers, rows)
    md += "\n_数值来自 plan §A.3 论文 v1.13 引用。复现完成后由 diff_with_paper L1 输出对照。_\n"
    return md


def build_table_5_4(main_records: list) -> tuple:
    """Table 5-4: 三重门槛通过率（UMC CV%, UAMCM CV%, shuffled-u 验证）。"""
    agg = aggregate_by_method_dataset(main_records, "ece", ddof=1)
    headers = ["Dataset", "UMC CV%", "UAMCM CV%", "UAMCM<UMC 改善"]
    rows = []
    csv_rows = []
    for ds in ("aliccp", "avazu", "criteo"):
        umc = agg.get((ds, "umc"))
        uamcm = agg.get((ds, "uamcm"))
        if not umc or not uamcm:
            rows.append([ds, "--", "--", "--"])
            csv_rows.append([ds, "", "", ""])
            continue
        umc_cv = cv_pct(umc["mean"], umc["std"])
        uamcm_cv = cv_pct(uamcm["mean"], uamcm["std"])
        improvement = uamcm["mean"] < umc["mean"]
        rows.append([
            ds, f"{umc_cv:.1f}%", f"{uamcm_cv:.1f}%",
            "Yes" if improvement else "No",
        ])
        csv_rows.append([ds, umc_cv, uamcm_cv, improvement])

    md = (
        "# Table 5-4: 三重门槛通过率\n\n"
        "门槛 1: UMC 在该数据集 ECE CV%（变异系数）\n"
        "门槛 2: UAMCM 在该数据集 ECE CV%\n"
        "门槛 3: UAMCM mean ECE < UMC mean ECE（改善方向）\n\n"
    )
    md += render_markdown_table(headers, rows)
    return md, ["dataset", "umc_cv_pct", "uamcm_cv_pct", "uamcm_better"], csv_rows


def build_table_5_5(v10_records: list) -> tuple:
    """Table 5-5: u_mode 三态对比（PE / shuffled / logit）。"""
    grouped: dict = {}
    for r in v10_records:
        key = (r["dataset"], r["u_mode"])
        grouped.setdefault(key, []).append(r["ece"])
    headers = ["Dataset", "PE ECE×100", "shuffled ECE×100", "logit ECE×100",
               "shuffled-PE 变化%"]
    rows = []
    csv_rows = []
    for ds in ("aliccp", "avazu", "criteo"):
        pe = grouped.get((ds, "pe"), [])
        shuf = grouped.get((ds, "shuffled"), [])
        log = grouped.get((ds, "logit"), [])
        pe_m = sum(pe) / len(pe) if pe else None
        shuf_m = sum(shuf) / len(shuf) if shuf else None
        log_m = sum(log) / len(log) if log else None
        change = (
            f"{100 * (shuf_m - pe_m) / pe_m:+.1f}%"
            if pe_m and shuf_m and pe_m != 0 else "--"
        )
        rows.append([
            ds,
            f"{pe_m*100:.2f}" if pe_m else "--",
            f"{shuf_m*100:.2f}" if shuf_m else "--",
            f"{log_m*100:.2f}" if log_m else "--",
            change,
        ])
        csv_rows.append([ds, pe_m, shuf_m, log_m, change])

    md = (
        "# Table 5-5: u_mode 消融（PE / shuffled / logit）\n\n"
        "u_mode=shuffled 时 u 与样本顺序完全解耦，理论上应消除 u 的方法学贡献。\n"
        "AliCCP/Criteo 期望显著恶化（u 有效）；Avazu 期望不显著恶化（u 无独立贡献）。\n\n"
    )
    md += render_markdown_table(headers, rows)
    return md, ["dataset", "pe_ece", "shuffled_ece", "logit_ece", "change_pct"], csv_rows


def build_table_5_6(v10_records: list) -> str:
    """Table 5-6: P5 论证支点 — 诊断预判 vs 实验方向命中率。"""
    grouped: dict = {}
    for r in v10_records:
        key = (r["dataset"], r["u_mode"])
        grouped.setdefault(key, []).append(r["ece"])
    headers = ["Dataset", "Ch3 诊断预判", "shuffled-u 实验结果", "方向命中"]
    rows = []
    predictions = {
        "aliccp": ("u 全域有效 → shuffled 应显著恶化", "supports"),
        "avazu": ("u 方向混乱 → shuffled 应不显著恶化", "supports"),
        "criteo": ("u 局部显著 → shuffled 应显著恶化", "supports"),
    }
    hits = 0
    total = 0
    for ds in ("aliccp", "avazu", "criteo"):
        pe = grouped.get((ds, "pe"), [])
        shuf = grouped.get((ds, "shuffled"), [])
        if not pe or not shuf:
            rows.append([ds, predictions[ds][0], "(no data)", "?"])
            continue
        pe_m = sum(pe) / len(pe)
        shuf_m = sum(shuf) / len(shuf)
        change_pct = 100 * (shuf_m - pe_m) / pe_m if pe_m else 0
        if ds == "avazu":
            hit = abs(change_pct) < 15
        else:
            hit = change_pct >= 30
        total += 1
        if hit:
            hits += 1
        rows.append([
            ds, predictions[ds][0],
            f"shuffled {change_pct:+.1f}%",
            "Yes" if hit else "No",
        ])

    md = (
        "# Table 5-6: P5 论证支点 — 诊断预判 = 实验验证 命中率\n\n"
        f"**命中率: {hits}/{total}** (论文期望 3/3)\n\n"
    )
    md += render_markdown_table(headers, rows)
    if total > 0:
        if hits == total:
            md += f"\n**结论**: 诊断预判 = 实验验证 全部命中，P5 论断成立。\n"
        else:
            md += f"\n**结论**: P5 仅 {hits}/{total} 命中，需 plan §M.4 根因分析。\n"
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str,
                    default=str(_PROJECT_ROOT / "results" / "tables"))
    args = ap.parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT))
    from reproduction.analysis.diff_with_paper import load_main_metrics, load_v10_metrics

    main_records = load_main_metrics()
    v10_records = load_v10_metrics()

    write_md(Path(args.out_dir) / "table_5_3.md", build_table_5_3())

    md54, h54, rows54 = build_table_5_4(main_records)
    write_md(Path(args.out_dir) / "table_5_4.md", md54)
    write_csv(Path(args.out_dir) / "table_5_4.csv", h54, rows54)

    md55, h55, rows55 = build_table_5_5(v10_records)
    write_md(Path(args.out_dir) / "table_5_5.md", md55)
    write_csv(Path(args.out_dir) / "table_5_5.csv", h55, rows55)

    write_md(Path(args.out_dir) / "table_5_6.md", build_table_5_6(v10_records))
    print(f"[tables_5_3_5_6] 4 tables written  "
          f"(main records={len(main_records)}, v10 records={len(v10_records)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
