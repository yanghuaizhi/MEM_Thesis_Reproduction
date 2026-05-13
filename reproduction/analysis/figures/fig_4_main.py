"""Ch4 图 4-1: 11 方法 × 3 数据集 ECE 柱状图（mean±std error bar）。

输入: experiments/runs/main/*/metrics.jsonl
输出: results/figures/fig_4_main.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import get_matplotlib, save_placeholder


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

METHODS = ["platt", "ir", "hb", "umnn", "neucalib", "desc", "sbcr",
           "umc", "umc_wor", "uamcm", "uamcm_wor"]
DATASETS = ["aliccp", "avazu", "criteo"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str,
                    default=str(_PROJECT_ROOT / "results" / "figures"))
    args = ap.parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT))
    from reproduction.analysis.diff_with_paper import load_main_metrics
    from reproduction.analysis.tables._common import aggregate_by_method_dataset

    out_pdf = Path(args.out_dir) / "fig_4_main.pdf"
    out_png = Path(args.out_dir) / "fig_4_main.png"

    records = load_main_metrics()
    if not records:
        save_placeholder(out_pdf, "fig_4_main:\nmain stage runs not yet completed.\nRun `orchestrator --stage main` first.")
        save_placeholder(out_png, "fig_4_main (no data)")
        print(f"[fig_4_main] placeholder written")
        return 0

    agg = aggregate_by_method_dataset(records, "ece", ddof=1)

    import numpy as np
    mpl, plt = get_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        means = []
        stds = []
        labels = []
        for m in METHODS:
            v = agg.get((ds, m))
            if v is None:
                continue
            means.append(v["mean"] * 100)            # ECE×100
            stds.append(v["std"] * 100)
            labels.append(m)
        if not means:
            ax.text(0.5, 0.5, f"{ds}: no data", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color="C0", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{ds}")
        ax.set_ylabel("ECE × 100 (M=100, ddof=1)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 4-1: ECE comparison across 11 methods × 3 datasets")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_4_main] {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
