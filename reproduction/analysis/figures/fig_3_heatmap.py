"""Ch3 图 3-2: AliCCP E[Y|p,u] 热力图（10 p-bins × 5 u-bins）。

输入: experiments/v9_samples/aliccp/*.npz
输出: results/figures/fig_3_heatmap.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import get_matplotlib, save_placeholder


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent


def compute_heatmap(y_pred, y_true, u, n_p_bins=10, n_u_bins=5):
    import numpy as np
    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    u = np.asarray(u, dtype=float)
    p_edges = np.quantile(yp, np.linspace(0, 1, n_p_bins + 1))
    u_edges = np.quantile(u, np.linspace(0, 1, n_u_bins + 1))
    p_edges[-1] += 1e-9
    u_edges[-1] += 1e-9
    p_ids = np.digitize(yp, p_edges[1:-1])
    u_ids = np.digitize(u, u_edges[1:-1])
    heat = np.full((n_p_bins, n_u_bins), np.nan)
    for i in range(n_p_bins):
        for j in range(n_u_bins):
            mask = (p_ids == i) & (u_ids == j)
            if mask.any():
                heat[i, j] = yt[mask].mean()
    return heat, p_edges, u_edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str,
                    default=str(_PROJECT_ROOT / "results" / "figures"))
    args = ap.parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT))
    from reproduction.analysis.diff_with_paper import load_v9_samples

    out_pdf = Path(args.out_dir) / "fig_3_heatmap.pdf"
    out_png = Path(args.out_dir) / "fig_3_heatmap.png"

    s = load_v9_samples("aliccp")
    if s is None or "y_pred_uncalib" not in s:
        save_placeholder(out_pdf, "fig_3_heatmap (aliccp):\nv9 samples not yet generated.")
        save_placeholder(out_png, "fig_3_heatmap (no data)")
        print(f"[fig_3_heatmap] placeholder written")
        return 0

    heat, p_edges, u_edges = compute_heatmap(s["y_pred_uncalib"], s["y_true"], s["u"])

    mpl, plt = get_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_title("Figure 3-2: AliCCP E[Y|p, u] heatmap (10 p-bins × 5 u-bins)")
    ax.set_xlabel("u quintile (low → high uncertainty)")
    ax.set_ylabel("p decile (low → high prediction)")
    fig.colorbar(im, ax=ax, label="E[Y | p-bin, u-bin]")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_3_heatmap] {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
