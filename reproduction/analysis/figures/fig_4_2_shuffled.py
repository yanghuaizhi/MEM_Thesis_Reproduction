"""Ch4 图 4-2: shuffled-u 消融对比（PE vs shuffled vs logit）。

输入: experiments/runs/v10/*/metrics.jsonl
输出: results/figures/fig_4_2_shuffled.{pdf,png}

布局: 1×3 子图（aliccp/avazu/criteo），分组柱状图（u_mode 颜色编码）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import get_matplotlib, save_placeholder


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

DATASETS = ["aliccp", "avazu", "criteo"]
U_MODES = ["pe", "shuffled", "logit"]
U_MODE_COLORS = {"pe": "C0", "shuffled": "C3", "logit": "C2"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str,
                    default=str(_PROJECT_ROOT / "results" / "figures"))
    args = ap.parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT))
    from reproduction.analysis.diff_with_paper import load_v10_metrics

    out_pdf = Path(args.out_dir) / "fig_4_2_shuffled.pdf"
    out_png = Path(args.out_dir) / "fig_4_2_shuffled.png"

    records = load_v10_metrics()
    if not records:
        save_placeholder(out_pdf, "fig_4_2_shuffled:\nv10 ablation runs not yet completed.\nRun `orchestrator --stage v10` first.")
        save_placeholder(out_png, "fig_4_2_shuffled (no data)")
        print(f"[fig_4_2_shuffled] placeholder written")
        return 0

    # 按 (dataset, u_mode) 聚合 mean ± std (ddof=1)
    grouped: dict = {}
    for r in records:
        key = (r["dataset"], r["u_mode"])
        grouped.setdefault(key, []).append(r.get("ece", 0))

    import math
    import numpy as np

    def stats(vals):
        if not vals:
            return None, None
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return m, 0.0
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        return m, s

    mpl, plt = get_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds in zip(axes, DATASETS):
        means = []
        stds = []
        for mode in U_MODES:
            m, s = stats(grouped.get((ds, mode), []))
            means.append((m or 0) * 100)
            stds.append((s or 0) * 100)
        x = np.arange(len(U_MODES))
        colors = [U_MODE_COLORS[m] for m in U_MODES]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(U_MODES)
        ax.set_title(f"{ds}")
        ax.set_ylabel("ECE × 100 (M=100, ddof=1)")
        ax.grid(axis="y", alpha=0.3)
        # 在 PE vs shuffled 之间画 ±σ 区域帮助判断 P4
        if grouped.get((ds, "pe")):
            pe_m, pe_s = stats(grouped[(ds, "pe")])
            if pe_m and pe_s:
                ax.axhspan((pe_m - pe_s) * 100, (pe_m + pe_s) * 100,
                          alpha=0.15, color="C0", label="PE ±σ")
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Figure 4-2: u_mode ablation (PE vs shuffled vs logit)")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_4_2_shuffled] {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
