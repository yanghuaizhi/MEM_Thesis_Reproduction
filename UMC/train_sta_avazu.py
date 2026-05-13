import os
import torch
import pandas as pd
import numpy as np
import random
import json


import sys

global root
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

# Path resolution: 30_reproduction project (see UMC/_paths.py)
from _paths import DATA_ROOT, CKPT_ROOT, setup_torch_uncertainty
setup_torch_uncertainty()

from models.inputs import SparseFeat
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metric import *
import torch.nn.functional as F
from utils.metric import *
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import LinearConstraint, Bounds
from time import time
LogisticRegression = None
try:
    import sklearn.utils.extmath as _sk_extmath

    if not hasattr(_sk_extmath, "log_logistic"):
        import numpy as _np

        def log_logistic(X, out=None):
            res = -_np.logaddexp(0.0, -X)
            if out is not None:
                out[...] = res
                return out
            return res

        _sk_extmath.log_logistic = log_logistic

    from clogistic import LogisticRegression as _ClogisticLogisticRegression

    LogisticRegression = _ClogisticLogisticRegression
except Exception:
    LogisticRegression = None


def setup_seed(seed):
    import torch  # # Warning: Do not remove, as it will cause an error later!!!

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def fit_parametric_calibrator(
    valid_logit_pred,
    valid_y,
    method,
    seed=1024,
    max_samples=-1,
    max_iter=200,
    constraint_weight=10.0,
    constraint_eps=1e-5,
    allow_subsample=False,
    max_constraint_weight=1e6,
    max_iter_cap=500,
    retries=6,
):
    if isinstance(valid_logit_pred, torch.Tensor):
        valid_logit_pred = valid_logit_pred.detach().cpu().numpy()
    if isinstance(valid_y, torch.Tensor):
        valid_y = valid_y.detach().cpu().numpy()

    z = np.asarray(valid_logit_pred).reshape(-1)
    y = np.asarray(valid_y).reshape(-1)

    n = int(z.shape[0])
    if n == 0:
        raise ValueError("empty calibration set")
    n_total = n

    rng = np.random.default_rng(int(seed))
    subsampled = False
    if max_samples is not None and int(max_samples) > 0 and n > int(max_samples):
        if not bool(allow_subsample):
            raise RuntimeError(
                f"subsample is forbidden in strict mode: total={n}, max_samples={int(max_samples)}"
            )
        idx = rng.choice(n, size=int(max_samples), replace=False)
        z = z[idx]
        y = y[idx]
        subsampled = True
        n = int(z.shape[0])

    z_t = torch.from_numpy(z.astype(np.float64)).reshape(-1)
    y_t = torch.from_numpy(y.astype(np.float64)).reshape(-1)

    if method == "platt":
        a_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)
        params = [a_raw, b]

        def logits_fn(z_in):
            a = F.softplus(a_raw)
            return a * z_in + b

        def penalty_fn():
            return torch.zeros((), dtype=torch.float64)
        shift = None

        def constraint_g_min():
            a = F.softplus(a_raw)
            return float(a.detach().cpu().item())

    elif method == "gauss":
        a_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
        b_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
        c = torch.zeros((), dtype=torch.float64, requires_grad=True)
        params = [a_raw, b_raw, c]
        smin, smax = -13.0, 13.0
        shift = None

        def logits_fn(z_in):
            a = F.softplus(a_raw)
            b = F.softplus(b_raw)
            return a * (z_in**2) + b * z_in + c

        def penalty_fn():
            a = F.softplus(a_raw)
            b = F.softplus(b_raw)
            g1 = 2.0 * a * float(smin) + b
            g2 = 2.0 * a * float(smax) + b
            return F.relu(-g1) ** 2 + F.relu(-g2) ** 2

        def constraint_g_min():
            a = F.softplus(a_raw)
            b = F.softplus(b_raw)
            g1 = 2.0 * a * float(smin) + b
            g2 = 2.0 * a * float(smax) + b
            return float(torch.minimum(g1, g2).detach().cpu().item())

    elif method == "gamma":
        a_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
        b_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
        c = torch.zeros((), dtype=torch.float64, requires_grad=True)
        params = [a_raw, b_raw, c]

        z_min = float(np.min(z))
        shift = max(13.0, -z_min + 1.0)
        smin, smax = 1e-4, 2.0 * shift

        def logits_fn(z_in):
            a = F.softplus(a_raw)
            b = F.softplus(b_raw)
            s = (z_in + shift).clamp_min(1e-12)
            return a * torch.log(s) + b * s + c

        def penalty_fn():
            a = F.softplus(a_raw)
            b = F.softplus(b_raw)
            g1 = a / float(smin) + b
            g2 = a / float(smax) + b
            return F.relu(-g1) ** 2 + F.relu(-g2) ** 2

        def constraint_g_min():
            a = F.softplus(a_raw)
            b = F.softplus(b_raw)
            g1 = a / float(smin) + b
            g2 = a / float(smax) + b
            return float(torch.minimum(g1, g2).detach().cpu().item())

    else:
        raise ValueError(f"unknown method: {method}")

    cur_weight = float(constraint_weight)
    cur_max_iter = int(max_iter)
    final_loss = None
    final_penalty = None
    g_min = None
    attempts = 0

    for attempt in range(int(retries)):
        attempts = attempt + 1
        opt = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=int(cur_max_iter),
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=20,
        )

        def closure():
            opt.zero_grad()
            logits = logits_fn(z_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            pen = penalty_fn()
            loss_total = loss + float(cur_weight) * pen
            loss_total.backward()
            return loss_total

        opt.step(closure)

        with torch.no_grad():
            logits = logits_fn(z_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            pen = penalty_fn()
            final_loss = float(loss.detach().cpu().item())
            final_penalty = float(pen.detach().cpu().item())
            g_min = float(constraint_g_min())

        if method == "platt":
            break
        if g_min >= -float(constraint_eps):
            break

        cur_weight = float(cur_weight) * 10.0
        cur_max_iter = min(int(max(1, round(cur_max_iter * 1.5))), int(max_iter_cap))
        if cur_weight > float(max_constraint_weight):
            break

    if method != "platt" and (g_min is None or g_min < -float(constraint_eps)):
        raise RuntimeError(
            f"constraint not satisfied: method={method}, g_min={g_min}, eps={float(constraint_eps)}, weight={cur_weight}, max_iter={cur_max_iter}"
        )

    if method == "platt":
        a = float(F.softplus(a_raw).detach().cpu().item())
        b_out = float(b.detach().cpu().item())
        return {
            "a": a,
            "b": b_out,
            "n_total": n_total,
            "n_used": n,
            "subsampled": subsampled,
            "final_calib_logloss": float(final_loss) if final_loss is not None else None,
            "final_penalty": float(final_penalty) if final_penalty is not None else None,
            "constraint_g_min": float(g_min) if g_min is not None else None,
            "constraint_eps": float(constraint_eps),
            "constraint_weight": float(cur_weight),
            "max_iter": int(cur_max_iter),
            "attempts": int(attempts),
        }
    if method == "gauss":
        a = float(F.softplus(a_raw).detach().cpu().item())
        b = float(F.softplus(b_raw).detach().cpu().item())
        c_out = float(c.detach().cpu().item())
        return {
            "a": a,
            "b": b,
            "c": c_out,
            "n_total": n_total,
            "n_used": n,
            "subsampled": subsampled,
            "final_calib_logloss": float(final_loss) if final_loss is not None else None,
            "final_penalty": float(final_penalty) if final_penalty is not None else None,
            "constraint_g_min": float(g_min) if g_min is not None else None,
            "constraint_eps": float(constraint_eps),
            "constraint_weight": float(cur_weight),
            "max_iter": int(cur_max_iter),
            "attempts": int(attempts),
        }
    a = float(F.softplus(a_raw).detach().cpu().item())
    b = float(F.softplus(b_raw).detach().cpu().item())
    c_out = float(c.detach().cpu().item())
    return {
        "a": a,
        "b": b,
        "c": c_out,
        "shift": float(shift),
        "n_total": n_total,
        "n_used": n,
        "subsampled": subsampled,
        "final_calib_logloss": float(final_loss) if final_loss is not None else None,
        "final_penalty": float(final_penalty) if final_penalty is not None else None,
        "constraint_g_min": float(g_min) if g_min is not None else None,
        "constraint_eps": float(constraint_eps),
        "constraint_weight": float(cur_weight),
        "max_iter": int(cur_max_iter),
        "attempts": int(attempts),
    }


class Config(object):
    def __init__(self):

        # Path resolution — resolved by 30_reproduction/UMC/_paths.py
        # Priority: MEM_DATA_ROOT / MEM_CKPT_ROOT env vars > 30_reproduction fallbacks
        self.data_root = DATA_ROOT
        self.data_name = "avazu"
        self.model_name = "deepfm"
        self.batch_size = 1024 * 32
        self.batch_size_calib = 1024 * 32
        self.dropout = 0.0
        self.init_std = 1e-4
        self.lr = 1e-3
        self.l2_reg = 0.0
        self.embedding_dim = 16
        self.num_estimators = 4
        self.alpha = 1.0
        self.gamma = 1
        self.seed = 1024
        self.filepath = CKPT_ROOT

        self.method = None
        self.field_index = 2
        self.num_workers = 8
        self.pin_memory = True
        self.persistent_workers = True
        self.logreg_solver = "scs"
        self.ece_M = 100
        self.hb_num_bins = 10
        self.sir_num_bins = 15
        self.sta_fit_max_samples = -1
        self.sta_allow_subsample = False
        self.sta_fit_max_iter = 200
        self.sta_fit_max_iter_cap = 500
        self.sta_constraint_weight = 10.0
        self.sta_constraint_weight_max = 1e6
        self.sta_constraint_eps = 1e-5
        self.sta_fit_retries = 6
        self.io_log_every = 200


def get_data(data_path=None):
    if not data_path:
        raise ValueError("data_path must be provided")

    path = os.path.join(data_path, "data.pkl")
    data = pd.read_pickle(filepath_or_buffer=path)

    feature_names = list(data.columns)[:-1]  # only sparse features
    label_names = list(data.columns)[-1]
    print("Feature names:", feature_names)
    print("Label names:", label_names)

    return data, feature_names, label_names


def trial(config_update):
    config = Config()
    if config_update is not None:
        for name, value in config_update.items():
            setattr(config, name, value)
    print(
        "config_update_json="
        + json.dumps(
            config_update if config_update is not None else {},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )

    experiment_name = ""
    experiment_info = [
        "data_name",
        "model_name",
        "batch_size",
        "dropout",
        "init_std",
        "lr",
        "l2_reg",
        "seed",
    ]
    if config.model_name == "packed_deepfm":
        experiment_info += ["num_estimators", "alpha", "gamma"]

    for name in experiment_info:
        value = getattr(config, name)
        experiment_name += name + "=" + str(value) + "_"

    print(
        f"run_start data_name={config.data_name} model_name={config.model_name} method={config.method}"
    )
    print(f"experiment_name={experiment_name}")
    setup_seed(1024)
    print("seed=1024")

    data, feature_names, label_names = get_data(
        data_path=os.path.join(config.data_root, config.data_name)
    )
    train, valid_test = train_test_split(
        data, test_size=0.4, random_state=1024, shuffle=False
    )
    valid, test = train_test_split(
        valid_test, test_size=0.5, random_state=1024, shuffle=False
    )
    feature_columns = [
        SparseFeat(
            feat,
            vocabulary_size=int(data[feat].max()) + 1,
            embedding_dim=config.embedding_dim,
        )
        for feat in feature_names
    ]

    valid_x = {name: np.array(valid[name]) for name in feature_names}
    valid_y = np.transpose([np.array(valid[label_names])])
    test_x = {name: np.array(test[name]) for name in feature_names}
    test_y = np.transpose([np.array(test[label_names])])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda_available={torch.cuda.is_available()}")
    print(f"split_sizes train={len(train)} valid={len(valid)} test={len(test)}")

    path = config.filepath + "/" + experiment_name + ".pth"
    print(f"ckpt_path={path}")
    model = torch.load(path, weights_only=False)
    model = model.to(device)
    # fix all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
    print(f"backbone_loaded model_class={model.__class__.__name__}")

    # valid and test dataloader
    if isinstance(valid_x, dict):
        valid_x = [valid_x[feature] for feature in model.feature_index]
    for i in range(len(valid_x)):
        if len(valid_x[i].shape) == 1:
            valid_x[i] = np.expand_dims(valid_x[i], axis=1)
    valid_tensor_data = Data.TensorDataset(
        torch.from_numpy(np.concatenate(valid_x, axis=-1)), torch.from_numpy(valid_y)
    )
    valid_loader = DataLoader(
        dataset=valid_tensor_data,
        shuffle=False,
        batch_size=config.batch_size_calib,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers and int(config.num_workers) > 0),
    )

    if isinstance(test_x, dict):
        test_x = [test_x[feature] for feature in model.feature_index]
    for i in range(len(test_x)):
        if len(test_x[i].shape) == 1:
            test_x[i] = np.expand_dims(test_x[i], axis=1)
    test_tensor_data = Data.TensorDataset(
        torch.from_numpy(np.concatenate(test_x, axis=-1)), torch.from_numpy(test_y)
    )
    test_loader = DataLoader(
        dataset=test_tensor_data,
        shuffle=False,
        batch_size=config.batch_size_calib,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers and int(config.num_workers) > 0),
    )
    print(
        f"dataloaders batch_size_calib={int(config.batch_size_calib)} valid_batches={len(valid_loader)} test_batches={len(test_loader)}"
    )

    # get valid y numpy list
    valid_y = []
    valid_y_pred = []
    valid_logit_pred = []
    io_log_every = int(getattr(config, "io_log_every", 2000))
    valid_t0 = time()
    print(f"collect_valid_predictions_start batches={len(valid_loader)}")
    for step, (x_valid, y_valid) in tqdm(enumerate(valid_loader), disable=1):
        x = x_valid.to(device).float()
        y = y_valid.to(device).float()
        valid_y.append(y.cpu().data.numpy())
        y_pred = model(x)
        eps = 1e-6
        y_pred = y_pred.clamp(eps, 1 - eps)
        valid_y_pred.append(y_pred.cpu().data.numpy())
        logit_pred = torch.logit(y_pred).cpu().data.numpy()
        valid_logit_pred.append(logit_pred)
        if io_log_every > 0 and (step + 1) % int(io_log_every) == 0:
            print(f"collect_valid_predictions_step step={step+1}/{len(valid_loader)}")
    valid_y = np.concatenate(valid_y).astype("float32").flatten()
    valid_y_pred = np.concatenate(valid_y_pred).astype("float32").flatten()
    valid_logit_pred = np.concatenate(valid_logit_pred).astype("float32").flatten()
    print(
        f"collect_valid_predictions_done seconds={time()-valid_t0:.2f} pred_mean={float(np.mean(valid_y_pred)):.8f} pred_std={float(np.std(valid_y_pred)):.8f} label_mean={float(np.mean(valid_y)):.8f}"
    )

    # evaluate function
    def evaluate(test_y_pred_calib, test_y, index):
        ece_M = int(getattr(config, "ece_M", 100))
        test_auc = get_auc(test_y, test_y_pred_calib)
        test_gauc = get_gauc(test_y, test_y_pred_calib, np.squeeze(test_x[index]))
        test_logloss = get_logloss(test_y, test_y_pred_calib)
        test_pcoc = get_pcoc(test_y, test_y_pred_calib)
        test_ece = get_ece(test_y, test_y_pred_calib, ece_M)
        test_fece = get_fece(test_y, test_y_pred_calib, np.squeeze(test_x[index]), 1)
        fece_list = get_mfece(test_y, test_y_pred_calib, test_x, 1)
        test_mfece = np.mean(fece_list)
        test_rce = get_rce(test_y, test_y_pred_calib, ece_M)
        test_frce = get_frce(test_y, test_y_pred_calib, np.squeeze(test_x[index]), 1)
        rce_list = get_mfrce(test_y, test_y_pred_calib, test_x, 1)
        test_mfrce = np.mean(rce_list)

        pred_mean = float(np.mean(test_y_pred_calib))
        pred_std = float(np.std(test_y_pred_calib))
        label_mean = float(np.mean(test_y))
        log = f"test_auc = {test_auc:.6f}, test_gauc = {test_gauc:.6f}, test_logloss = {test_logloss:.6f}, test_pcoc = {test_pcoc:.6f}, test_ece = {test_ece:.6f}, test_fece = {test_fece:.6f}, test_mfece = {test_mfece:.6f}, test_rce = {test_rce:.6f}, test_frce = {test_frce:.6f}, test_mfrce = {test_mfrce:.6f}"
        fece_log = f"multi_field_fece_list={np.asarray(fece_list, dtype=np.float64).round(8).tolist()}"
        rce_log = f"multi_field_rce_list={np.asarray(rce_list, dtype=np.float64).round(8).tolist()}"
        dist_log = (
            f"pred_mean={pred_mean:.8f}, pred_std={pred_std:.8f}, label_mean={label_mean:.8f}, ece_M={ece_M}"
        )
        print(log)
        print(dist_log)
        print(fece_log)
        print(rce_log)

    # different methods, based on binning or distribution
    test_sigma2 = []
    if config.method == "hb":
        num_bins = int(getattr(config, "hb_num_bins", 10))
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        confidences = valid_y_pred
        indices = np.digitize(confidences, bins, right=True)
        bin_accuracies = np.zeros(num_bins, dtype=np.float32)
        bin_counts = np.zeros(num_bins, dtype=np.int64)
        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(valid_y.flatten()[selected])
                bin_counts[b] = int(len(selected))
        non_empty = int(np.sum(bin_counts > 0))
        if non_empty > 0:
            acc_nonempty = bin_accuracies[bin_counts > 0]
            print(
                f"hb_bins num_bins={int(num_bins)} non_empty={non_empty} acc_min={float(np.min(acc_nonempty)):.8f} acc_max={float(np.max(acc_nonempty)):.8f}"
            )
        print(
            f"credibility_summary method=hb status=PASS calib_used={int(len(valid_y))} calib_total={int(len(valid_y))} subsampled=False"
        )

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_t0 = time()
        print(f"collect_test_predictions_start batches={len(test_loader)}")
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader), disable=1):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_sigma2.append(model.sigma2_epistemic.cpu().data.numpy())
                test_y_pred.append(y_pred.cpu().data.numpy())
                indices = np.digitize(y_pred.cpu().data.numpy(), bins, right=True)
                y_pred_calib = bin_accuracies[indices - 1]
                test_y_pred_calib.append(y_pred_calib)
                if io_log_every > 0 and (step + 1) % int(io_log_every) == 0:
                    print(f"collect_test_predictions_step step={step+1}/{len(test_loader)}")
        print(f"collect_test_predictions_done seconds={time()-test_t0:.2f}")

    elif config.method == "ir":
        model_calib = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        model_calib.fit(valid_y_pred, valid_y)
        print(
            f"credibility_summary method=ir status=PASS calib_used={int(len(valid_y))} calib_total={int(len(valid_y))} subsampled=False"
        )

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_t0 = time()
        print(f"collect_test_predictions_start batches={len(test_loader)}")
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader), disable=1):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_sigma2.append(model.sigma2_epistemic.cpu().data.numpy())
                test_y_pred.append(y_pred.cpu().data.numpy())
                y_pred_calib = model_calib.predict(y_pred.cpu().data.numpy())
                test_y_pred_calib.append(y_pred_calib)
                if io_log_every > 0 and (step + 1) % int(io_log_every) == 0:
                    print(f"collect_test_predictions_step step={step+1}/{len(test_loader)}")
        print(f"collect_test_predictions_done seconds={time()-test_t0:.2f}")

    elif config.method == "sir":
        num_bins = int(getattr(config, "sir_num_bins", 15))
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        confidences = valid_y_pred.flatten()
        indices = np.digitize(confidences, bins, right=True)
        bin_accuracies = np.zeros(num_bins, dtype=np.float32)
        bin_count = np.zeros(num_bins, dtype=np.float32)
        bin_min = np.zeros(num_bins, dtype=np.float32)
        bin_max = np.zeros(num_bins, dtype=np.float32)
        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(valid_y.flatten()[selected])
                bin_count[b] = len(selected)
                bin_min[b] = np.min(confidences.flatten()[selected])
                bin_max[b] = np.max(confidences.flatten()[selected])
        bin_mid = (bin_min + bin_max) / 2

        print(
            f"sir_bins_init num_bins={int(num_bins)} non_empty={int(np.sum(bin_count > 0))} count_min={int(np.min(bin_count))} count_max={int(np.max(bin_count))}"
        )
        i = 0
        while i < len(bin_mid) - 1:
            if bin_accuracies[i] > bin_accuracies[i + 1]:
                bin_mid[i] = (
                    bin_count[i] * bin_mid[i] + bin_count[i + 1] * bin_mid[i + 1]
                ) / (bin_count[i] + bin_count[i + 1])
                bin_accuracies[i] = (
                    bin_count[i] * bin_accuracies[i]
                    + bin_count[i + 1] * bin_accuracies[i + 1]
                ) / (bin_count[i] + bin_count[i + 1])
                bin_mid = np.delete(bin_mid, i + 1, 0)
                bin_accuracies = np.delete(bin_accuracies, i + 1, 0)
                bin_count = np.delete(bin_count, i + 1, 0)
            else:
                i = i + 1

        print(f"sir_bins_final effective_bins={int(len(bin_mid))}")
        bin_mid = np.insert(np.append(bin_mid, 1.0), 0, 0.0)
        bin_accuracies = np.insert(np.append(bin_accuracies, 1.0), 0, 0.0)
        weight = np.diff(bin_accuracies) / (np.diff(bin_mid) + 1e-4)
        bias = bin_accuracies[1:] - weight * bin_mid[1:]

        assert np.all(np.diff(bin_accuracies) >= 0)
        print(
            f"credibility_summary method=sir status=PASS calib_used={int(len(valid_y))} calib_total={int(len(valid_y))} subsampled=False"
        )

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_t0 = time()
        print(f"collect_test_predictions_start batches={len(test_loader)}")
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader), disable=1):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_sigma2.append(model.sigma2_epistemic.cpu().data.numpy())
                test_y_pred.append(y_pred.cpu().data.numpy())
                indices = np.digitize(y_pred.cpu().data.numpy(), bin_mid, right=True)
                y_pred_calib = (
                    weight[indices - 1] * y_pred.cpu().data.numpy() + bias[indices - 1]
                )
                test_y_pred_calib.append(y_pred_calib)
                if io_log_every > 0 and (step + 1) % int(io_log_every) == 0:
                    print(f"collect_test_predictions_step step={step+1}/{len(test_loader)}")
        print(f"collect_test_predictions_done seconds={time()-test_t0:.2f}")

    elif config.method == "platt":
        fit = fit_parametric_calibrator(
            valid_logit_pred,
            valid_y,
            "platt",
            seed=int(config.seed),
            max_samples=int(getattr(config, "sta_fit_max_samples", -1)),
            max_iter=int(getattr(config, "sta_fit_max_iter", 200)),
            constraint_weight=float(getattr(config, "sta_constraint_weight", 10.0)),
            constraint_eps=float(getattr(config, "sta_constraint_eps", 1e-5)),
            allow_subsample=bool(getattr(config, "sta_allow_subsample", False)),
            max_constraint_weight=float(
                getattr(config, "sta_constraint_weight_max", 1e6)
            ),
            max_iter_cap=int(getattr(config, "sta_fit_max_iter_cap", 500)),
            retries=int(getattr(config, "sta_fit_retries", 6)),
        )
        print(
            f"credibility_summary method=platt status=PASS calib_used={fit['n_used']} calib_total={fit['n_total']} subsampled={fit['subsampled']} calib_logloss={fit['final_calib_logloss']} g_min={fit['constraint_g_min']} eps={fit['constraint_eps']} weight={fit['constraint_weight']} max_iter={fit['max_iter']} attempts={fit['attempts']}"
        )
        a = torch.tensor(fit["a"]).to(device)
        b = torch.tensor(fit["b"]).to(device)

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_t0 = time()
        print(f"collect_test_predictions_start batches={len(test_loader)}")
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader), disable=1):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_sigma2.append(model.sigma2_epistemic.cpu().data.numpy())
                eps = 1e-6
                y_pred = y_pred.clamp(eps, 1 - eps)
                test_y_pred.append(y_pred.cpu().data.numpy())
                logit_calib = a * torch.logit(y_pred) + b
                y_pred_calib = F.sigmoid(logit_calib).cpu().data.numpy()
                test_y_pred_calib.append(y_pred_calib)
                if io_log_every > 0 and (step + 1) % int(io_log_every) == 0:
                    print(f"collect_test_predictions_step step={step+1}/{len(test_loader)}")
        print(f"collect_test_predictions_done seconds={time()-test_t0:.2f}")

    elif config.method == "gauss":
        fit = fit_parametric_calibrator(
            valid_logit_pred,
            valid_y,
            "gauss",
            seed=int(config.seed),
            max_samples=int(getattr(config, "sta_fit_max_samples", -1)),
            max_iter=int(getattr(config, "sta_fit_max_iter", 200)),
            constraint_weight=float(getattr(config, "sta_constraint_weight", 10.0)),
            constraint_eps=float(getattr(config, "sta_constraint_eps", 1e-5)),
            allow_subsample=bool(getattr(config, "sta_allow_subsample", False)),
            max_constraint_weight=float(
                getattr(config, "sta_constraint_weight_max", 1e6)
            ),
            max_iter_cap=int(getattr(config, "sta_fit_max_iter_cap", 500)),
            retries=int(getattr(config, "sta_fit_retries", 6)),
        )
        print(
            f"credibility_summary method=gauss status=PASS calib_used={fit['n_used']} calib_total={fit['n_total']} subsampled={fit['subsampled']} calib_logloss={fit['final_calib_logloss']} g_min={fit['constraint_g_min']} eps={fit['constraint_eps']} weight={fit['constraint_weight']} max_iter={fit['max_iter']} attempts={fit['attempts']}"
        )
        a = torch.tensor(fit["a"]).to(device)
        b = torch.tensor(fit["b"]).to(device)
        c = torch.tensor(fit["c"]).to(device)

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_t0 = time()
        print(f"collect_test_predictions_start batches={len(test_loader)}")
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader), disable=1):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_sigma2.append(model.sigma2_epistemic.cpu().data.numpy())
                eps = 1e-6
                y_pred = y_pred.clamp(eps, 1 - eps)
                test_y_pred.append(y_pred.cpu().data.numpy())
                logit_calib = (
                    a * torch.pow(torch.logit(y_pred), 2) + b * torch.logit(y_pred) + c
                )
                y_pred_calib = F.sigmoid(logit_calib).cpu().data.numpy()
                test_y_pred_calib.append(y_pred_calib)
                if io_log_every > 0 and (step + 1) % int(io_log_every) == 0:
                    print(f"collect_test_predictions_step step={step+1}/{len(test_loader)}")
        print(f"collect_test_predictions_done seconds={time()-test_t0:.2f}")

    elif config.method == "gamma":
        fit = fit_parametric_calibrator(
            valid_logit_pred,
            valid_y,
            "gamma",
            seed=int(config.seed),
            max_samples=int(getattr(config, "sta_fit_max_samples", -1)),
            max_iter=int(getattr(config, "sta_fit_max_iter", 200)),
            constraint_weight=float(getattr(config, "sta_constraint_weight", 10.0)),
            constraint_eps=float(getattr(config, "sta_constraint_eps", 1e-5)),
            allow_subsample=bool(getattr(config, "sta_allow_subsample", False)),
            max_constraint_weight=float(
                getattr(config, "sta_constraint_weight_max", 1e6)
            ),
            max_iter_cap=int(getattr(config, "sta_fit_max_iter_cap", 500)),
            retries=int(getattr(config, "sta_fit_retries", 6)),
        )
        print(
            f"credibility_summary method=gamma status=PASS calib_used={fit['n_used']} calib_total={fit['n_total']} subsampled={fit['subsampled']} calib_logloss={fit['final_calib_logloss']} g_min={fit['constraint_g_min']} eps={fit['constraint_eps']} weight={fit['constraint_weight']} max_iter={fit['max_iter']} attempts={fit['attempts']}"
        )
        a = torch.tensor(fit["a"]).to(device)
        b = torch.tensor(fit["b"]).to(device)
        c = torch.tensor(fit["c"]).to(device)
        shift = float(fit["shift"])

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_t0 = time()
        print(f"collect_test_predictions_start batches={len(test_loader)}")
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader), disable=1):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_sigma2.append(model.sigma2_epistemic.cpu().data.numpy())
                eps = 1e-6
                y_pred = y_pred.clamp(eps, 1 - eps)
                test_y_pred.append(y_pred.cpu().data.numpy())
                logit_calib = (
                    a * torch.log((torch.logit(y_pred) + shift).clamp_min(1e-12))
                    + b * (torch.logit(y_pred) + shift)
                    + c
                )
                y_pred_calib = F.sigmoid(logit_calib).cpu().data.numpy()
                test_y_pred_calib.append(y_pred_calib)
                if io_log_every > 0 and (step + 1) % int(io_log_every) == 0:
                    print(f"collect_test_predictions_step step={step+1}/{len(test_loader)}")
        print(f"collect_test_predictions_done seconds={time()-test_t0:.2f}")

    else:
        return NotImplementedError("Not implement")

    test_y = np.concatenate(test_y).astype("float32").flatten()
    test_y_pred = np.concatenate(test_y_pred).astype("float32").flatten()
    test_y_pred_calib = np.concatenate(test_y_pred_calib).astype("float32").flatten()
    test_sigma2 = np.concatenate(test_sigma2).astype("float32").flatten()
    print("metrics_tag=uncalibrated")
    evaluate(test_y_pred, test_y, config.field_index)
    print("metrics_tag=calibrated")
    evaluate(test_y_pred_calib, test_y, config.field_index)
    if getattr(config, "uncertainty_bin_eval", False):
        n_bins = int(getattr(config, "uncertainty_bin_num_bins", 20))
        ece_M = int(getattr(config, "uncertainty_bin_ece_M", 100))
        df = get_uncertainty_bin_table(
            test_y,
            test_y_pred_calib,
            test_sigma2,
            n_bins=n_bins,
            eps=1e-8,
            use_log_sigma2=True,
            ece_M=ece_M,
        )
        print("uncertainty_bin_table")
        print(df.to_string(index=False))
        save_path = getattr(config, "uncertainty_bin_save_path", None)
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(
                f"uncertainty_bin_saved path={save_path} rows={int(df.shape[0])} cols={int(df.shape[1])}"
            )


if __name__ == "__main__":

    config_update = {
        "data_name": "avazu",
        "model_name": "packed_deepfm",
        "batch_size": 1024 * 32,
        "dropout": 0.1,
        "init_std": 1e-4,
        "lr": 5e-4,
        "l2_reg": 1e-5,
        "seed": 1024,
        "num_estimators": 16,
        "alpha": 1.0,
        "gamma": 1,
        "method": "sir",
        "field_index": 2,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "logreg_solver": "scs",
    }
    trial(config_update=config_update)
