"""Ch3 图 3-1~3-4: 三数据集 PCOC-u 分位分布（4 子图）。

输入: experiments/v9_samples/<dataset>/*.npz （含 y_pred_uncalib, y_true, u）
输出: results/figures/fig_3_pcoc_u_dist.{pdf,png}

布局: 1×3 子图（aliccp/avazu/criteo），x=u 分位桶 (1-20)，y=PCOC。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import get_matplotlib, save_placeholder


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

DATASETS = ["aliccp", "avazu", "criteo"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str,
                    default=str(_PROJECT_ROOT / "results" / "figures"))
    args = ap.parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT))
    from reproduction.analysis.diff_with_paper import (
        load_v9_samples, compute_per_u_bin_pcoc,
    )

    samples = {ds: load_v9_samples(ds) for ds in DATASETS}
    have_data = any(s is not None for s in samples.values())

    out_pdf = Path(args.out_dir) / "fig_3_pcoc_u_dist.pdf"
    out_png = Path(args.out_dir) / "fig_3_pcoc_u_dist.png"

    if not have_data:
        save_placeholder(out_pdf, "fig_3_pcoc_u_dist:\nv9 samples NPZ not yet generated.\nRun `orchestrator --stage v9` first.")
        save_placeholder(out_png, "fig_3_pcoc_u_dist (no data)")
        print(f"[fig_3_pcoc_u_dist] placeholder written to {out_pdf}")
        return 0

    mpl, plt = get_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ds in zip(axes, DATASETS):
        s = samples[ds]
        if s is None or "y_pred_uncalib" not in s:
            ax.text(0.5, 0.5, f"{ds}: no data", ha="center", va="center")
            ax.set_axis_off()
            continue
        per_bin = compute_per_u_bin_pcoc(
            s["y_pred_uncalib"], s["y_true"], s["u"], n_bins=20
        )
        bins = per_bin["bins"]
        pcoc = per_bin["pcoc_per_bin"]
        ax.plot(bins, pcoc, marker="o", color="C0")
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6, label="PCOC=1")
        ax.set_title(f"{ds} — PCOC per u-bin (over_pred bins={per_bin['over_predict_bins']}/20)")
        ax.set_xlabel("u quantile bin")
        ax.set_ylabel("PCOC = E[y_pred]/E[y_true]")
        ax.set_xticks(list(range(0, 21, 5)))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Figure 3-1: PCOC-u distribution across datasets (Ch3 diagnosis)")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_3_pcoc_u_dist] {out_pdf}\n[fig_3_pcoc_u_dist] {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
