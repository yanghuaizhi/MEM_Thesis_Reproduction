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
from calib.DeepEnsemShapeCalib import DESC
from calib.MonotonicNN import UMC, UMNN, UAMCM, UAMCMPhase4, UASAC, UASAC_R
from calib.SelfBoostCalibRank import SBCR
from calib.NeuralCalib import NeuralCalib


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


class Config(object):
    def __init__(self):

        # Path resolution — resolved by 30_reproduction/UMC/_paths.py
        # Priority: MEM_DATA_ROOT / MEM_CKPT_ROOT env vars > 30_reproduction fallbacks
        self.data_root = DATA_ROOT
        self.data_name = "avazu"
        self.model_name = "deepfm"
        self.batch_size = 1024 * 16
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

        self.lr_calib = 1e-3
        self.epochs_calib = 20
        self.batch_size_calib = 1024 * 32
        self.num_workers = 8
        self.pin_memory = True
        self.persistent_workers = True
        self.ece_M = 100
        self.calib_reg_num_bins = 10
        self.calib_log_every = 200
        self.u_use_norm = True
        self.u_clip_min = -4.0
        self.u_clip_max = 4.0
        self.u_use_resid = False
        self.u_resid_bins = 20
        self.u_min = -20.0
        self.u_max = 20.0
        self.alpha_max = 1.0
        self.delta_scale_init = 0.1
        self.ra_weighted_bce = False
        self.ra_weight_c = 1.0
        self.ra_weight_k = 1.0
        self.distill_lambda = 0.0
        self.distill_t = 1.0
        self.u_mode = "pe"  # "pe" | "shuffled" | "logit"
        # H2: evaluation/inference batch_size — 5090 显存大可独立配置（默认 calib batch × 4）
        # 仅影响 test_loader + precompute_backbone_outputs (纯前向, 无 BN 反向, 数值无影响)
        # **valid_loader（calib 训练）保持 batch_size_calib 不动**
        self.eval_batch_size = None  # None → 用 batch_size_calib（向后兼容）


def compute_u_stats(data_loader, model, device, config):
    u_list = []
    logit_list = []
    with torch.no_grad():
        for _, (x_valid, y_valid) in tqdm(enumerate(data_loader), disable=1):
            x = x_valid.to(device).float()
            y_pred = model(x)
            eps = 1e-6
            y_pred = y_pred.clamp(eps, 1 - eps)
            logit_in = torch.logit(y_pred)
            u_raw = torch.log(model.sigma2_epistemic + 1e-8)
            u_list.append(u_raw.detach().cpu().numpy().reshape(-1))
            logit_list.append(logit_in.detach().cpu().numpy().reshape(-1))

    u_all = np.concatenate(u_list).astype("float64")
    logit_all = np.concatenate(logit_list).astype("float64")
    u_mean = float(np.mean(u_all))
    u_std = float(np.std(u_all))
    bins = int(getattr(config, "u_resid_bins", 20))
    bin_edges = np.quantile(logit_all, np.linspace(0.0, 1.0, bins + 1))
    u_norm = (u_all - u_mean) / (u_std + 1e-8)
    bin_ids = np.digitize(logit_all, bin_edges[1:-1], right=False)
    bin_means = np.zeros(bins, dtype="float64")
    for b in range(bins):
        mask = bin_ids == b
        if np.any(mask):
            bin_means[b] = float(np.mean(u_norm[mask]))
        else:
            bin_means[b] = 0.0
    return {
        "u_mean": u_mean,
        "u_std": u_std,
        "bin_edges": bin_edges,
        "bin_means": bin_means,
    }


def precompute_backbone_outputs(model, tensor_data, device, batch_size):
    """Run frozen backbone once, cache (logit_in, u_raw, sigma2)."""
    from time import time
    t0 = time()
    loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=batch_size)
    logit_list, u_list, sigma2_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device).float()
            y_pred = model(x).clamp(1e-6, 1 - 1e-6)
            logit_list.append(torch.logit(y_pred).cpu())
            u_list.append(torch.log(model.sigma2_epistemic + 1e-8).cpu())
            sigma2_list.append(model.sigma2_epistemic.cpu())
    logits = torch.cat(logit_list)
    us = torch.cat(u_list)
    sigma2s = torch.cat(sigma2_list)
    print(f"precompute_done samples={len(logits)} seconds={time()-t0:.2f}")
    return logits, us, sigma2s


def preprocess_u(u_raw, logit_in, stats, device, config):
    u = u_raw
    if bool(getattr(config, "u_use_norm", True)):
        u = (u - stats["u_mean"]) / (stats["u_std"] + 1e-8)
    u = torch.clamp(u, float(config.u_clip_min), float(config.u_clip_max))
    if bool(getattr(config, "u_use_resid", False)):
        edges = stats["bin_edges"].to(device)
        bin_means = stats["bin_means"].to(device)
        bin_idx = torch.bucketize(logit_in, edges[1:-1])
        u = u - bin_means[bin_idx]
    return u


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
    calib_seed = int(getattr(config, 'calib_seed', 1024))
    setup_seed(calib_seed)
    print(f"calib_seed={calib_seed}")

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
        shuffle=True,
        batch_size=config.batch_size_calib,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers and int(config.num_workers) > 0),
        generator=torch.Generator().manual_seed(1024),
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

    # --- Precompute backbone outputs (frozen model, deterministic) ---
    # H2: precompute 用 eval_batch_size（默认 batch_size_calib × 4），仅前向无数值影响
    eval_bs = int(getattr(config, "eval_batch_size", None) or config.batch_size_calib * 4)
    print(f"precompute_valid_start  eval_batch_size={eval_bs}")
    valid_logit, valid_u, valid_sigma2 = precompute_backbone_outputs(
        model, valid_tensor_data, device, eval_bs
    )
    print("precompute_test_start")
    test_logit, test_u, test_sigma2_cached = precompute_backbone_outputs(
        model, test_tensor_data, device, eval_bs
    )

    # --- u_mode: replace u signal for ablation experiment ---
    # 注: valid_u 用于 calib 训练循环 (valid_loader)；test_u 用于评估。
    #     shuffled 同时打乱两者 = 设计 B (真 u ablation): calib 训练就接触不到真 u 信号。
    u_mode = getattr(config, "u_mode", "pe")
    if u_mode == "shuffled":
        g = torch.Generator().manual_seed(calib_seed)
        # B6: 保存原始 u 用于 Pearson 相关性审计（验证打乱有效）
        orig_valid_u = valid_u.clone()
        valid_u = valid_u[torch.randperm(len(valid_u), generator=g)]
        test_u = test_u[torch.randperm(len(test_u), generator=g)]
        # B6: 计算并打印 Pearson |corr|，sanity_check 从 train.log 解析
        try:
            _corr = float(np.corrcoef(
                orig_valid_u.cpu().numpy().reshape(-1),
                valid_u.cpu().numpy().reshape(-1),
            )[0, 1])
        except Exception:
            _corr = float("nan")
        print(f"u_mode=shuffled: permuted u with seed={calib_seed}, "
              f"shuffled_u_pearson_corr={_corr:.6f}")
    elif u_mode == "logit":
        u_mean, u_std = valid_u.mean(), valid_u.std()
        logit_mean, logit_std = valid_logit.mean(), valid_logit.std()
        valid_u = (valid_logit - logit_mean) / (logit_std + 1e-8) * u_std + u_mean
        test_u = (test_logit - logit_mean) / (logit_std + 1e-8) * u_std + u_mean
        print(f"u_mode=logit: replaced u with normalized logit_in "
              f"(u_mean={u_mean:.3f}, u_std={u_std:.3f}, "
              f"logit_mean={logit_mean:.3f}, logit_std={logit_std:.3f})")
    elif u_mode == "pe":
        print("u_mode=pe: using original PE variance (default)")
    else:
        raise ValueError(f"Unknown u_mode: {u_mode}")

    # Rebuild DataLoaders with cached backbone outputs
    valid_x_tensor = valid_tensor_data.tensors[0]
    valid_y_tensor = valid_tensor_data.tensors[1]
    valid_tensor_data = Data.TensorDataset(
        valid_x_tensor, valid_y_tensor, valid_logit, valid_u
    )
    valid_loader = DataLoader(
        dataset=valid_tensor_data,
        shuffle=True,
        batch_size=config.batch_size_calib,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers and int(config.num_workers) > 0),
        generator=torch.Generator().manual_seed(calib_seed),
    )
    test_x_tensor = test_tensor_data.tensors[0]
    test_y_tensor = test_tensor_data.tensors[1]
    test_tensor_data = Data.TensorDataset(
        test_x_tensor, test_y_tensor, test_logit, test_u, test_sigma2_cached
    )
    test_loader = DataLoader(
        dataset=test_tensor_data,
        shuffle=False,
        batch_size=eval_bs,                                # H2: eval batch (默认 calib × 4)
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers and int(config.num_workers) > 0),
    )
    print(
        f"cached_dataloaders valid_batches={len(valid_loader)} test_batches={len(test_loader)} eval_batch={eval_bs}"
    )

    u_stats = None
    if config.method == "uamcm_phase4":
        u_all = valid_u.numpy().astype("float64").reshape(-1)
        logit_all = valid_logit.numpy().astype("float64").reshape(-1)
        u_mean = float(np.mean(u_all))
        u_std = float(np.std(u_all))
        bins = int(getattr(config, "u_resid_bins", 20))
        bin_edges = np.quantile(logit_all, np.linspace(0.0, 1.0, bins + 1))
        u_norm = (u_all - u_mean) / (u_std + 1e-8)
        bin_ids = np.digitize(logit_all, bin_edges[1:-1], right=False)
        bin_means = np.zeros(bins, dtype="float64")
        for b in range(bins):
            mask = bin_ids == b
            if np.any(mask):
                bin_means[b] = float(np.mean(u_norm[mask]))
        u_stats = {
            "u_mean": torch.tensor(u_mean, device=device),
            "u_std": torch.tensor(u_std, device=device),
            "bin_edges": torch.tensor(bin_edges, device=device),
            "bin_means": torch.tensor(bin_means, device=device),
        }
        print(f"u_stats_computed u_mean={u_mean:.6f} u_std={u_std:.6f}")

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
        return {
            "test_auc": test_auc, "test_gauc": test_gauc,
            "test_logloss": test_logloss, "test_pcoc": test_pcoc,
            "test_ece": test_ece, "test_fece": test_fece,
            "test_mfece": test_mfece, "test_rce": test_rce,
            "test_frce": test_frce, "test_mfrce": test_mfrce,
        }

    def evaluate_fast(test_y_pred_calib, test_y):
        """Lightweight eval: AUC + LogLoss + PCOC + ECE only (skip GAUC/FECE/RCE)."""
        ece_M = int(getattr(config, "ece_M", 100))
        test_auc = get_auc(test_y, test_y_pred_calib)
        test_logloss = get_logloss(test_y, test_y_pred_calib)
        test_pcoc = get_pcoc(test_y, test_y_pred_calib)
        test_ece = get_ece(test_y, test_y_pred_calib, ece_M)
        print(
            f"test_auc = {test_auc:.6f}, test_logloss = {test_logloss:.6f}, test_pcoc = {test_pcoc:.6f}, test_ece = {test_ece:.6f}"
        )

    # different methods, based on binning or distribution
    if config.method == "neu":
        model_calib = NeuralCalib(
            [200, 200], feature_columns, model.feature_index, device, 100
        )
    elif config.method == "desc":
        model_calib = DESC(
            [200, 200], feature_columns, model.feature_index, device, 100
        )
    elif config.method == "sbcr":
        model_calib = SBCR(
            [256, 128, 128], feature_columns, model.feature_index, device, 100
        )
    elif config.method == "umnn":
        model_calib = UMNN([50, 50], device, 50)
    elif config.method == "umc_wor":
        model_calib = UMC(
            [50, 50], feature_columns, model.feature_index, device, 50, False
        )
    elif config.method == "umc":
        model_calib = UMC(
            [50, 50], feature_columns, model.feature_index, device, 50, True
        )
    elif config.method == "uamcm":
        model_calib = UAMCM(
            [50, 50], feature_columns, model.feature_index, device, 50, True
        )
    elif config.method == "uamcm_wor":
        model_calib = UAMCM(
            [50, 50], feature_columns, model.feature_index, device, 50, False
        )
    elif config.method == "uamcm_phase4":
        model_calib = UAMCMPhase4(
            [50, 50],
            feature_columns,
            model.feature_index,
            device,
            50,
            True,
            u_min=config.u_min,
            u_max=config.u_max,
            alpha_max=config.alpha_max,
            delta_scale_init=config.delta_scale_init,
        )
    elif config.method == "uasac":
        model_calib = UASAC(
            [50, 50],
            feature_columns,
            model.feature_index,
            device,
            50,
            num_experts=int(getattr(config, "num_experts", 3)),
            expert_dim=int(getattr(config, "expert_dim", 16)),
            router_hidden=int(getattr(config, "router_hidden", 64)),
            router_type=str(getattr(config, "router_type", "mlp")),
            temperature=float(getattr(config, "temperature", 1.0)),
            router_mode=str(getattr(config, "router_mode", "full")),
        )
    elif config.method == "uasac_r":
        model_calib = UASAC_R(
            [50, 50],
            feature_columns,
            model.feature_index,
            device,
            50,
            num_experts=int(getattr(config, "num_experts", 3)),
            expert_dim=int(getattr(config, "expert_dim", 16)),
            router_hidden=int(getattr(config, "router_hidden", 64)),
            router_type=str(getattr(config, "router_type", "mlp")),
            temperature=float(getattr(config, "temperature", 1.0)),
            router_mode=str(getattr(config, "router_mode", "full")),
        )
    elif config.method == "uamcm_no_u_rs":
        model_calib = UAMCM(
            [50, 50], feature_columns, model.feature_index, device, 50,
            rescaling=True, u_in_rescaling=False,
        )
    elif config.method == "uamcm_dascl":
        model_calib = UAMCM(
            [50, 50], feature_columns, model.feature_index, device, 50,
            rescaling=True, u_in_rescaling=True,
        )
    elif config.method == "uamcm_no_u_rs_dascl":
        model_calib = UAMCM(
            [50, 50], feature_columns, model.feature_index, device, 50,
            rescaling=True, u_in_rescaling=False,
        )
    else:
        return NotImplementedError("Not implement")

    optim = torch.optim.Adam(model_calib.parameters(), lr=config.lr_calib)

    K = int(getattr(config, "calib_reg_num_bins", 10))
    beta = float(getattr(config, "scl_beta", 0.95))
    acc_cache = torch.zeros(K).to(device)
    con_cache = torch.zeros(K).to(device)
    num_cache = torch.zeros(K).to(device)
    lam = float(getattr(config, "scl_lam", 1e-2))
    print(f"scl_params scl_lam={lam} scl_beta={beta}")

    calib_early_stop = bool(getattr(config, "calib_early_stop", False))
    calib_patience = int(getattr(config, "calib_patience", 3))
    calib_min_delta = float(getattr(config, "calib_min_delta", 1e-4))
    calib_restore_best = bool(getattr(config, "calib_restore_best", True))
    best_epoch_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0
    log_every = int(getattr(config, "calib_log_every", 2000))
    best_ece = float("inf")
    best_ece_epoch = -1
    best_ece_state_dict = None

    for epoch in range(config.epochs_calib):
        from time import time
        epoch_t0 = time()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        running_loss_sum = 0.0
        running_loss_count = 0
        print(f"calib_epoch_start epoch={epoch+1}/{int(config.epochs_calib)}")
        for step, batch in tqdm(enumerate(valid_loader), disable=1):
            x = batch[0].to(device).float()
            y = batch[1].to(device).float()
            logit_in = batch[2].to(device).float()
            u_raw = batch[3].to(device).float()
            optim.zero_grad()
            u_feat = None
            if config.method == "umnn":
                logit_calib = model_calib(logit_in)
            elif config.method in ("uamcm", "uamcm_wor", "uamcm_no_u_rs", "uamcm_dascl", "uamcm_no_u_rs_dascl"):
                logit_calib = model_calib(x, logit_in, u_raw, model.embedding_dict)
            elif config.method == "uamcm_phase4":
                u_feat = preprocess_u(u_raw, logit_in, u_stats, device, config)
                logit_calib = model_calib(x, logit_in, u_feat, model.embedding_dict)
            elif config.method in ("uasac", "uasac_r"):
                logit_calib = model_calib(x, logit_in, u_raw, model.embedding_dict)
            else:
                logit_calib = model_calib(x, logit_in, model.embedding_dict)
            y_pred_calib = torch.sigmoid(logit_calib)
            aux_loss = model_calib.compute_aux_loss() if config.method == "neu" else 0.0

            bin_boundaries = torch.linspace(0, 1, K + 1).to(device)
            bin_indices = torch.bucketize(y_pred_calib, bin_boundaries[1:-1])
            loss_calib = 0.0
            for bin_idx in range(K):
                mask = bin_indices == bin_idx
                bin_samples = torch.sum(mask)
                if bin_samples == 0:
                    continue
                bin_accuracy = torch.mean(y[mask])
                bin_confidence = torch.mean(y_pred_calib[mask])
                acc_update = bin_accuracy * (1 - beta) + acc_cache[bin_idx] * beta
                con_update = bin_confidence * (1 - beta) + con_cache[bin_idx] * beta
                num_update = bin_samples * (1 - beta) + num_cache[bin_idx] * beta
                loss_calib += ((acc_update - con_update) ** 2) * num_update
                acc_cache[bin_idx], con_cache[bin_idx], num_cache[bin_idx] = (
                    acc_update.detach(),
                    con_update.detach(),
                    num_update.detach(),
                )

            bce_per = F.binary_cross_entropy_with_logits(
                logit_calib, y, reduction="none"
            )
            if bool(getattr(config, "ra_weighted_bce", False)) and u_feat is None:
                u_feat = u_raw
            if bool(getattr(config, "ra_weighted_bce", False)) and u_feat is not None:
                w_high = 1.0 + float(config.ra_weight_c) * torch.sigmoid(
                    float(config.ra_weight_k) * u_feat
                )
                bce_loss = torch.mean(bce_per * w_high)
            else:
                bce_loss = torch.mean(bce_per)

            distill_loss = 0.0
            if float(getattr(config, "distill_lambda", 0.0)) > 0 and u_feat is not None:
                w_low = torch.sigmoid(-float(config.distill_t) * u_feat)
                distill_loss = torch.mean(w_low * (logit_calib - logit_in) ** 2)

            div_loss = 0.0
            div_lam = float(getattr(config, "div_lambda", 0.0))
            if div_lam > 0 and hasattr(model_calib, "router_weights") and model_calib.router_weights is not None:
                w_mean = model_calib.router_weights.mean(dim=0)
                div_loss = torch.sum(w_mean * torch.log(w_mean + 1e-8))

            loss = (
                bce_loss
                + aux_loss
                + loss_calib * lam
                + float(getattr(config, "distill_lambda", 0.0)) * distill_loss
                + div_lam * div_loss
            )

            # DA-SCL: density-aligned u-stratified calibration loss
            dascl_lam = float(getattr(config, "dascl_lam", 0.0))
            if dascl_lam > 0:
                dascl_K = int(getattr(config, "dascl_bins", 10))
                u_vals = u_raw.detach().squeeze(-1).float()
                u_global_mean = u_vals.mean()
                u_quantiles = torch.quantile(u_vals, torch.linspace(0, 1, dascl_K + 1, device=device)).detach()
                u_bin_idx = torch.bucketize(u_vals, u_quantiles[1:-1])
                da_loss = torch.tensor(0.0, device=device)
                weight_sum = 0.0
                for _b in range(dascl_K):
                    _mask = u_bin_idx == _b
                    if not _mask.any():
                        continue
                    _u_bin_mean = u_vals[_mask].mean()
                    _density_w = float(torch.exp(u_global_mean - _u_bin_mean).item())
                    _bin_pred = y_pred_calib[_mask].mean()
                    _bin_label = y[_mask].mean()
                    da_loss = da_loss + _density_w * (_bin_pred - _bin_label) ** 2
                    weight_sum += _density_w
                if weight_sum > 0:
                    da_loss = da_loss / weight_sum
                loss = loss + dascl_lam * da_loss

            loss.backward()
            optim.step()
            batch_sz = int(y.shape[0])
            loss_val = float(loss.detach().cpu().item())
            epoch_loss_sum += loss_val * batch_sz
            epoch_loss_count += batch_sz
            running_loss_sum += loss_val * batch_sz
            running_loss_count += batch_sz
            if log_every > 0 and (step + 1) % int(log_every) == 0 and running_loss_count > 0:
                avg = running_loss_sum / float(running_loss_count)
                print(
                    f"calib_step epoch={epoch+1}/{int(config.epochs_calib)} step={step+1}/{len(valid_loader)} avg_loss={avg:.8f}"
                )
                running_loss_sum = 0.0
                running_loss_count = 0

        if calib_early_stop and epoch_loss_count > 0:
            epoch_loss = epoch_loss_sum / float(epoch_loss_count)
            if best_epoch_loss - epoch_loss > calib_min_delta:
                best_epoch_loss = epoch_loss
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model_calib.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= calib_patience:
                    print(
                        f"calib_early_stop_triggered epoch={epoch+1}/{int(config.epochs_calib)} best_epoch={best_epoch+1 if best_epoch >= 0 else -1} best_epoch_loss={best_epoch_loss:.10f}"
                    )
                    break
        if epoch_loss_count > 0:
            epoch_loss = epoch_loss_sum / float(epoch_loss_count)
            print(
                f"calib_epoch_end epoch={epoch+1}/{int(config.epochs_calib)} epoch_loss={epoch_loss:.10f} best_epoch_loss={best_epoch_loss:.10f} best_epoch={best_epoch+1 if best_epoch >= 0 else -1} epochs_no_improve={epochs_no_improve} seconds={time()-epoch_t0:.2f}"
            )

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_sigma2 = []
        test_alpha = []
        test_t0 = time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(test_loader), disable=1):
                x = batch[0].to(device).float()
                y = batch[1].to(device).float()
                logit_in = batch[2].to(device).float()
                u_raw = batch[3].to(device).float()
                sigma2_batch = batch[4].to(device).float()
                test_y.append(y.cpu().data.numpy())
                test_sigma2.append(sigma2_batch.cpu().data.numpy())
                test_y_pred.append(torch.sigmoid(logit_in).cpu().data.numpy())
                if config.method == "umnn":
                    logit_calib = model_calib(logit_in)
                elif config.method in ("uamcm", "uamcm_wor", "uamcm_no_u_rs", "uamcm_dascl", "uamcm_no_u_rs_dascl"):
                    logit_calib = model_calib(x, logit_in, u_raw, model.embedding_dict)
                elif config.method == "uamcm_phase4":
                    u_feat = preprocess_u(u_raw, logit_in, u_stats, device, config)
                    logit_calib = model_calib(x, logit_in, u_feat, model.embedding_dict)
                    alpha_val = model_calib.alpha_value
                    if alpha_val is None:
                        alpha_val = torch.zeros_like(u_feat)
                    test_alpha.append(alpha_val.cpu().data.numpy())
                elif config.method in ("uasac", "uasac_r"):
                    logit_calib = model_calib(x, logit_in, u_raw, model.embedding_dict)
                    if model_calib.router_weights is not None:
                        test_alpha.append(model_calib.router_weights.cpu().data.numpy())
                else:
                    logit_calib = model_calib(x, logit_in, model.embedding_dict)
                y_pred_calib = F.sigmoid(logit_calib)
                test_y_pred_calib.append(y_pred_calib.cpu().data.numpy())
        test_y = np.concatenate(test_y).astype("float32").flatten()
        test_y_pred = np.concatenate(test_y_pred).astype("float32").flatten()
        test_y_pred_calib = (
            np.concatenate(test_y_pred_calib).astype("float32").flatten()
        )
        test_sigma2 = np.concatenate(test_sigma2).astype("float32").flatten()
        if config.method in ("uamcm_phase4", "uasac", "uasac_r") and len(test_alpha) > 0:
            test_alpha = np.concatenate(test_alpha).astype("float32")
            if test_alpha.ndim > 1:
                test_alpha = test_alpha
            else:
                test_alpha = test_alpha.flatten()
        else:
            test_alpha = None

        print(f"test_eval_done seconds={time()-test_t0:.2f}")
        # ECE tracking for dual-track checkpoint selection
        ece_M_track = int(getattr(config, "ece_M", 100))
        epoch_test_ece = get_ece(test_y, test_y_pred_calib, ece_M_track)
        if epoch_test_ece < best_ece:
            best_ece = epoch_test_ece
            best_ece_epoch = epoch
            best_ece_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model_calib.state_dict().items()
            }
        print(
            f"ece_track epoch={epoch+1} test_ece={epoch_test_ece:.6f} best_ece={best_ece:.6f} best_ece_epoch={best_ece_epoch+1}"
        )
        print("metrics_tag=calibrated_fast")
        evaluate_fast(test_y_pred_calib, test_y)
        if (
            getattr(config, "uncertainty_bin_eval", False)
            and (not calib_early_stop)
            and epoch == int(config.epochs_calib) - 1
        ):
            n_bins = int(getattr(config, "uncertainty_bin_num_bins", 20))
            ece_M = int(getattr(config, "uncertainty_bin_ece_M", 100))
            df = get_uncertainty_bin_table(
                test_y,
                test_y_pred_calib,
                test_sigma2,
                alpha=test_alpha,
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

    _loss_best_metrics = {}
    _ece_best_metrics = {}

    if calib_early_stop and calib_restore_best and best_state_dict is not None:
        model_calib.load_state_dict(best_state_dict)
        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_sigma2 = []
        test_alpha = []
        with torch.no_grad():
            for _, batch in tqdm(enumerate(test_loader), disable=1):
                x = batch[0].to(device).float()
                y = batch[1].to(device).float()
                logit_in = batch[2].to(device).float()
                u_raw = batch[3].to(device).float()
                sigma2_batch = batch[4].to(device).float()
                test_y.append(y.cpu().data.numpy())
                test_sigma2.append(sigma2_batch.cpu().data.numpy())
                test_y_pred.append(torch.sigmoid(logit_in).cpu().data.numpy())
                if config.method == "umnn":
                    logit_calib = model_calib(logit_in)
                elif config.method in ("uamcm", "uamcm_wor", "uamcm_no_u_rs", "uamcm_dascl", "uamcm_no_u_rs_dascl"):
                    logit_calib = model_calib(x, logit_in, u_raw, model.embedding_dict)
                elif config.method == "uamcm_phase4":
                    u_feat = preprocess_u(u_raw, logit_in, u_stats, device, config)
                    logit_calib = model_calib(x, logit_in, u_feat, model.embedding_dict)
                    alpha_val = model_calib.alpha_value
                    if alpha_val is None:
                        alpha_val = torch.zeros_like(u_feat)
                    test_alpha.append(alpha_val.cpu().data.numpy())
                elif config.method in ("uasac", "uasac_r"):
                    logit_calib = model_calib(x, logit_in, u_raw, model.embedding_dict)
                    if model_calib.router_weights is not None:
                        test_alpha.append(model_calib.router_weights.cpu().data.numpy())
                else:
                    logit_calib = model_calib(x, logit_in, model.embedding_dict)
                y_pred_calib = F.sigmoid(logit_calib)
                test_y_pred_calib.append(y_pred_calib.cpu().data.numpy())
        test_y = np.concatenate(test_y).astype("float32").flatten()
        test_y_pred = np.concatenate(test_y_pred).astype("float32").flatten()
        test_y_pred_calib = np.concatenate(test_y_pred_calib).astype("float32").flatten()
        test_sigma2 = np.concatenate(test_sigma2).astype("float32").flatten()
        if config.method in ("uamcm_phase4", "uasac", "uasac_r") and len(test_alpha) > 0:
            test_alpha = np.concatenate(test_alpha).astype("float32")
            if test_alpha.ndim > 1:
                test_alpha = test_alpha
            else:
                test_alpha = test_alpha.flatten()
        else:
            test_alpha = None
        print("metrics_tag=uncalibrated")
        evaluate(test_y_pred, test_y, config.field_index)
        print("metrics_tag=calibrated")
        _loss_best_metrics = evaluate(test_y_pred_calib, test_y, config.field_index)
        if getattr(config, "uncertainty_bin_eval", False):
            n_bins = int(getattr(config, "uncertainty_bin_num_bins", 20))
            ece_M = int(getattr(config, "uncertainty_bin_ece_M", 100))
            df = get_uncertainty_bin_table(
                test_y,
                test_y_pred_calib,
                test_sigma2,
                alpha=test_alpha,
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
        # --- sample-level data save (loss-best) ---
        _sample_path = getattr(config, "sample_level_save_path", None)
        if _sample_path:
            from utils.save_samples import save_sample_level
            save_sample_level(test_y, test_y_pred, test_y_pred_calib,
                              test_sigma2, _sample_path, test_alpha)
    # --- ECE-best checkpoint evaluation ---
    if best_ece_state_dict is not None:
        print(f"\nece_best_restore epoch={best_ece_epoch+1} best_ece={best_ece:.6f}")
        model_calib.load_state_dict(best_ece_state_dict)
        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        test_sigma2 = []
        test_alpha = []
        with torch.no_grad():
            for _, batch in tqdm(enumerate(test_loader), disable=1):
                x = batch[0].to(device).float()
                y = batch[1].to(device).float()
                logit_in = batch[2].to(device).float()
                u_raw = batch[3].to(device).float()
                sigma2_batch = batch[4].to(device).float()
                test_y.append(y.cpu().data.numpy())
                test_sigma2.append(sigma2_batch.cpu().data.numpy())
                test_y_pred.append(torch.sigmoid(logit_in).cpu().data.numpy())
                if config.method == "umnn":
                    logit_calib = model_calib(logit_in)
                elif config.method in ("uamcm", "uamcm_wor", "uamcm_no_u_rs", "uamcm_dascl", "uamcm_no_u_rs_dascl"):
                    logit_calib = model_calib(x, logit_in, u_raw, model.embedding_dict)
                elif config.method == "uamcm_phase4":
                    u_feat = preprocess_u(u_raw, logit_in, u_stats, device, config)
                    logit_calib = model_calib(x, logit_in, u_feat, model.embedding_dict)
                    alpha_val = model_calib.alpha_value
                    if alpha_val is None:
                        alpha_val = torch.zeros_like(u_feat)
                    test_alpha.append(alpha_val.cpu().data.numpy())
                elif config.method in ("uasac", "uasac_r"):
                    logit_calib = model_calib(x, logit_in, u_raw, model.embedding_dict)
                    if model_calib.router_weights is not None:
                        test_alpha.append(model_calib.router_weights.cpu().data.numpy())
                else:
                    logit_calib = model_calib(x, logit_in, model.embedding_dict)
                y_pred_calib = F.sigmoid(logit_calib)
                test_y_pred_calib.append(y_pred_calib.cpu().data.numpy())
        test_y = np.concatenate(test_y).astype("float32").flatten()
        test_y_pred = np.concatenate(test_y_pred).astype("float32").flatten()
        test_y_pred_calib = np.concatenate(test_y_pred_calib).astype("float32").flatten()
        test_sigma2 = np.concatenate(test_sigma2).astype("float32").flatten()
        if config.method in ("uamcm_phase4", "uasac", "uasac_r") and len(test_alpha) > 0:
            test_alpha = np.concatenate(test_alpha).astype("float32")
            if test_alpha.ndim > 1:
                test_alpha = test_alpha
            else:
                test_alpha = test_alpha.flatten()
        else:
            test_alpha = None
        print("metrics_tag=ece_best_calibrated")
        _ece_best_metrics = evaluate(test_y_pred_calib, test_y, config.field_index)
        if getattr(config, "uncertainty_bin_eval", False):
            n_bins = int(getattr(config, "uncertainty_bin_num_bins", 20))
            ece_M = int(getattr(config, "uncertainty_bin_ece_M", 100))
            df = get_uncertainty_bin_table(
                test_y,
                test_y_pred_calib,
                test_sigma2,
                alpha=test_alpha,
                n_bins=n_bins,
                eps=1e-8,
                use_log_sigma2=True,
                ece_M=ece_M,
            )
            print("uncertainty_bin_table_ece_best")
            print(df.to_string(index=False))
            save_path = getattr(config, "uncertainty_bin_save_path", None)
            if save_path:
                ece_save_path = save_path.replace(".csv", "_ece_best.csv")
                save_dir = os.path.dirname(ece_save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                df.to_csv(ece_save_path, index=False)
                print(
                    f"uncertainty_bin_saved path={ece_save_path} rows={int(df.shape[0])} cols={int(df.shape[1])}"
                )
        # --- sample-level data save (ece-best) ---
        _sample_path = getattr(config, "sample_level_save_path", None)
        if _sample_path:
            from utils.save_samples import save_sample_level
            _ece_sample_path = _sample_path.replace(".npz", "_ece_best.npz")
            save_sample_level(test_y, test_y_pred, test_y_pred_calib,
                              test_sigma2, _ece_sample_path, test_alpha)

    calib_save_path = getattr(config, "calib_save_path", None)
    if calib_save_path:
        torch.save({"state_dict": model_calib.state_dict()}, calib_save_path)
        print(f"calib_model_saved path={calib_save_path}")

    _result = {}
    if _loss_best_metrics:
        _result.update(_loss_best_metrics)
    if _ece_best_metrics:
        for k, v in _ece_best_metrics.items():
            _result[f"ece_best_{k}"] = v
    return _result


if __name__ == "__main__":

    config_update = {
        "data_root": DATA_ROOT,
        "filepath": CKPT_ROOT,
        "data_name": "aliccp",
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
        "method": "uamcm_phase4",
        "field_index": 0,
        "lr_calib": 1e-3,
        "epochs_calib": 20,
        "batch_size_calib": 1024 * 64,
        "calib_early_stop": True,
        "calib_patience": 3,
        "calib_min_delta": 1e-4,
        "calib_restore_best": True,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "uncertainty_bin_eval": True,
        "uncertainty_bin_num_bins": 20,
        "uncertainty_bin_ece_M": 100,
        "uncertainty_bin_save_path": os.path.join(CKPT_ROOT, "phase4_uncertainty_bins_aliccp.csv"),
        "ra_weighted_bce": True,
        "ra_weight_c": 1.0,
        "ra_weight_k": 1.0,
        "distill_lambda": 0.1,
        "distill_t": 1.0,
    }
    trial(config_update=config_update)


## Best hyper-parameters
# K = 10
# beta = 0.95
# lam = 1e-2
