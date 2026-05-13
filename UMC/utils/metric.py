import pandas as pd
import numpy as np
from sklearn.metrics import *


def get_gauc(y_true, y_pred, field_id):
    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "field_id": field_id})
    grouped_data = data.groupby("field_id")

    group_aucs = []
    group_sizes = []

    for field_id, group in grouped_data:
        group_y_true = group["y_true"].values
        group_y_pred = group["y_pred"].values
        if 0 < np.mean(group_y_true) < 1:
            auc = roc_auc_score(group_y_true, group_y_pred)
            group_aucs.append(auc)
            group_sizes.append(len(group))

    group_aucs = np.array(group_aucs)
    group_sizes = np.array(group_sizes)

    gauc = np.average(group_aucs, weights=group_sizes)
    return gauc


def get_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc


def get_logloss(y_true, y_pred):
    logloss = log_loss(y_true, y_pred)
    return logloss


def get_pcoc(y_true, y_pred):
    pcoc = np.sum(y_pred) / np.sum(y_true)
    return pcoc


def get_ece(y_true, y_pred, M=100):
    data = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "bin_id": (y_pred * M).astype("int32")}
    )
    q_curve = data.groupby("bin_id").agg(
        {
            "y_true": ["mean", "count"],
            "y_pred": ["mean"],
        }
    )

    ece = np.sum(
        np.abs(q_curve["y_true"]["mean"] - q_curve["y_pred"]["mean"])
        * q_curve["y_true"]["count"]
    )
    n = np.sum(q_curve["y_true"]["count"])
    return ece / n


def get_rce(y_true, y_pred, M=100):
    data = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "bin_id": (y_pred * M).astype("int32")}
    )
    q_curve = data.groupby("bin_id").agg(
        {
            "y_true": ["mean", "count"],
            "y_pred": ["mean"],
        }
    )
    # remove zeros
    q_curve = q_curve[q_curve["y_true"]["mean"] > 0]
    rce = np.sum(
        np.abs(q_curve["y_true"]["mean"] - q_curve["y_pred"]["mean"])
        * q_curve["y_true"]["count"]
        / (q_curve["y_true"]["mean"])
    )

    n = np.sum(q_curve["y_true"]["count"])
    return rce / n


def get_fece(y_true, y_pred, field_id, M=1):
    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "bin_id": (y_pred * M).astype("int32"),
            "field_id": field_id,
        }
    )
    q_curve = data.groupby(["field_id", "bin_id"], as_index=True).agg(
        {
            "y_true": ["mean", "count"],
            "y_pred": ["mean"],
        }
    )
    fece = np.sum(
        np.abs(q_curve["y_true"]["mean"] - q_curve["y_pred"]["mean"])
        * q_curve["y_true"]["count"]
    )
    n = np.sum(q_curve["y_true"]["count"])
    return fece / n


def get_frce(y_true, y_pred, field_id, M=1):
    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "bin_id": (y_pred * M).astype("int32"),
            "field_id": field_id,
        }
    )
    q_curve = data.groupby(["field_id", "bin_id"], as_index=True).agg(
        {
            "y_true": ["mean", "count"],
            "y_pred": ["mean"],
        }
    )
    # remove zeros
    q_curve = q_curve[q_curve["y_true"]["mean"] > 0]
    frce = np.sum(
        np.abs(q_curve["y_true"]["mean"] - q_curve["y_pred"]["mean"])
        * q_curve["y_true"]["count"]
        / (q_curve["y_true"]["mean"])
    )

    n = np.sum(q_curve["y_true"]["count"])
    return frce / n


def get_mfece(y_true, y_pred, x, M=1):
    result = []
    for field_id in x:
        field_id = np.squeeze(field_id)
        result.append(get_fece(y_true, y_pred, field_id, M))
    return np.array(result)


def get_mfrce(y_true, y_pred, x, M=1):
    result = []
    for field_id in x:
        field_id = np.squeeze(field_id)
        result.append(get_frce(y_true, y_pred, field_id, M))
    return np.array(result)


def get_uncertainty_bin_table(
    y_true,
    y_pred,
    sigma2,
    alpha=None,
    n_bins=20,
    eps=1e-8,
    use_log_sigma2=True,
    ece_M=100,
):
    y_true = np.asarray(y_true).astype("float64").reshape(-1)
    y_pred = np.asarray(y_pred).astype("float64").reshape(-1)
    sigma2 = np.asarray(sigma2).astype("float64").reshape(-1)
    alpha_arr = None
    alpha_2d = None
    if alpha is not None:
        alpha_np = np.asarray(alpha).astype("float64")
        if alpha_np.ndim == 2:
            alpha_2d = alpha_np  # (N, K) router weights
            alpha_arr = None
        else:
            alpha_arr = alpha_np.reshape(-1)

    if use_log_sigma2:
        u = np.log(sigma2 + eps)
    else:
        u = sigma2

    if len(u) == 0:
        extra_cols = []
        if alpha_2d is not None:
            extra_cols = [f"w_{k}" for k in range(alpha_2d.shape[1])]
        elif alpha_arr is not None:
            extra_cols = ["alpha_mean", "alpha_std"]
        return pd.DataFrame(
            columns=[
                "binning",
                "n_bins_req",
                "n_bins_eff",
                "bin_id",
                "count",
                "count_ratio",
                "u_min",
                "u_max",
                "u_mean",
                "u_std",
                "sigma2_mean",
                "sigma2_std",
                "y_true_mean",
                "y_true_std",
                "y_true_sum",
                "y_pred_mean",
                "y_pred_std",
                "y_pred_sum",
                "pcoc",
                "logloss",
                "ece",
            ]
            + extra_cols
        )

    if np.nanstd(u) < 1e-12:
        bin_id = np.zeros_like(u, dtype="int64")
        binning = "degenerate_all_zero"
    else:
        try:
            bin_id = pd.qcut(u, q=int(n_bins), labels=False, duplicates="drop")
            bin_id = np.asarray(bin_id)
            bin_id = np.nan_to_num(bin_id, nan=0.0).astype("int64")
            binning = "qcut_quantile"
        except Exception:
            bin_id = np.zeros_like(u, dtype="int64")
            binning = "qcut_failed_all_zero"

    data_dict = {
        "y_true": y_true,
        "y_pred": y_pred,
        "sigma2": sigma2,
        "u": u,
        "bin_id": bin_id,
    }
    if alpha_arr is not None:
        data_dict["alpha"] = alpha_arr
    if alpha_2d is not None:
        for k in range(alpha_2d.shape[1]):
            data_dict[f"w_{k}"] = alpha_2d[:, k]
    data = pd.DataFrame(data_dict)

    n_total = int(len(data))
    n_bins_req = int(n_bins)
    n_bins_eff = int(len(pd.unique(data["bin_id"])))

    rows = []
    for bid, g in data.groupby("bin_id", sort=True):
        y_t = g["y_true"].values
        y_p = g["y_pred"].values
        sigma2_g = g["sigma2"].values
        u_g = g["u"].values
        y_true_sum = float(np.sum(y_t))
        y_pred_sum = float(np.sum(y_p))
        row = {
                "binning": str(binning),
                "n_bins_req": int(n_bins_req),
                "n_bins_eff": int(n_bins_eff),
                "bin_id": int(bid),
                "count": int(len(g)),
                "count_ratio": float(len(g)) / float(n_total) if n_total > 0 else float("nan"),
                "u_min": float(np.min(u_g)),
                "u_max": float(np.max(u_g)),
                "u_mean": float(np.mean(u_g)),
                "u_std": float(np.std(u_g)),
                "sigma2_mean": float(np.mean(sigma2_g)),
                "sigma2_std": float(np.std(sigma2_g)),
                "y_true_mean": float(np.mean(y_t)),
                "y_true_std": float(np.std(y_t)),
                "y_true_sum": float(y_true_sum),
                "y_pred_mean": float(np.mean(y_p)),
                "y_pred_std": float(np.std(y_p)),
                "y_pred_sum": float(y_pred_sum),
                "pcoc": float(y_pred_sum / y_true_sum) if y_true_sum > 0 else float("nan"),
                "logloss": float(get_logloss(y_t, y_p)),
                "ece": float(get_ece(y_t, y_p, M=ece_M)),
        }
        if alpha_2d is not None:
            num_experts = alpha_2d.shape[1]
            for k in range(num_experts):
                row[f"w_{k}"] = float(np.mean(g[f"w_{k}"].values))
        elif alpha_arr is not None:
            row["alpha_mean"] = float(np.mean(g["alpha"].values))
            row["alpha_std"] = float(np.std(g["alpha"].values))
        rows.append(row)

    return pd.DataFrame(rows)
