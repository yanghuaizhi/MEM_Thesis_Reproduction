import os
import pandas as pd
import torch
import numpy as np
import random
import torch.backends

import sys
import datetime

global root
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

# Path resolution: 30_reproduction project (see UMC/_paths.py)
from _paths import DATA_ROOT, CKPT_ROOT, setup_torch_uncertainty
setup_torch_uncertainty()

from models.callbacks import EarlyStopping, ModelCheckpoint
from models.inputs import SparseFeat
from models.deepfm import DeepFM, PackedDeepFM
from sklearn.metrics import *
try:
    from ray import tune
except Exception:
    tune = None
from sklearn.model_selection import train_test_split
from utils.metric import *


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


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()



class Config(object):
    def __init__(self):
        # data — resolved by 30_reproduction/UMC/_paths.py
        # Priority: MEM_DATA_ROOT env var > <30_reproduction>/data/processed fallback
        self.data_root = DATA_ROOT
        self.data_name = "avazu"
        self.field_index = 2

        # model
        self.model_name = "packed_deepfm"
        self.l2_reg = 1e-6
        self.init_std = 1e-4
        self.hidden_units = [512, 256, 128, 64]
        self.dropout = 0.1
        self.embedding_dim = 16
        self.num_estimators = 8
        self.alpha = 1.0
        self.gamma = 1

        # train/test
        self.seed = 1024
        self.batch_size = 1024 * 32
        self.epochs = 20
        self.optim = "adam"
        self.lr = 1e-3
        self.loss = "bce"
        self.metrics = ["auc"]
        self.verbose = 2  #  0 = silent, 1 = progress bar, 2 = one line per epoch.
        self.use_tune = False  # use ray.tune or not]
        self.shuffle = False
        self.num_workers = 8
        self.pin_memory = True
        self.persistent_workers = True

        # earlystopping
        self.monitor = "val_auc"
        self.min_delta = 1e-5
        self.patience = 3
        self.mode = "max"
        self.restore_best_weights = True

        # modelcheckpoint — resolved by 30_reproduction/UMC/_paths.py
        # Priority: MEM_CKPT_ROOT env var > <30_reproduction>/experiments fallback
        self.filepath = CKPT_ROOT
        self.save_best_only = True
        self.save_weights_only = False
        self.save_freq = "epoch"
        self.is_save = True

        # history
        self.history_path = os.path.join(root, "history")


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


def log_packed_uncertainty_stats(model, x_dict, feature_names, batch_size, device):
    x_list = [x_dict[name] for name in feature_names]
    for i in range(len(x_list)):
        if len(x_list[i].shape) == 1:
            x_list[i] = np.expand_dims(x_list[i], axis=1)
    x_np = np.concatenate(x_list, axis=-1)
    x_batch = torch.from_numpy(x_np[:batch_size]).float().to(device)
    with torch.no_grad():
        y_pred = model(x_batch)
        sigma2 = model.sigma2_epistemic

    p_mean = y_pred.detach().cpu().numpy()
    sigma2_np = sigma2.detach().cpu().numpy()
    p_min = float(np.min(p_mean))
    p_max = float(np.max(p_mean))
    sigma2_mean = float(np.mean(sigma2_np))
    sigma2_std = float(np.std(sigma2_np))
    q50, q90, q99 = np.quantile(sigma2_np, [0.5, 0.9, 0.99])
    zero_ratio = float(np.mean(sigma2_np == 0))

    print("PackedDeepFM sanity stats")
    print("p_mean_min", p_min, "p_mean_max", p_max)
    print(
        "sigma2_mean",
        sigma2_mean,
        "sigma2_std",
        sigma2_std,
        "sigma2_p50",
        float(q50),
        "sigma2_p90",
        float(q90),
        "sigma2_p99",
        float(q99),
        "sigma2_zero_ratio",
        zero_ratio,
    )


def trial(config_update):
    config = Config()
    if config_update is not None:
        for name, value in config_update.items():
            setattr(config, name, value)

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

    run_basepath = os.path.join(config.filepath, experiment_name)
    log_path = run_basepath + ".log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = TeeStream(orig_stdout, log_f)
    sys.stderr = TeeStream(orig_stderr, log_f)

    print(
        "run_start",
        datetime.datetime.now().isoformat(timespec="seconds"),
        "log_path",
        log_path,
    )

    setup_seed(config.seed)

    try:
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

        train_x = {name: np.array(train[name]) for name in feature_names}
        train_y = np.transpose([np.array(train[label_names])])
        valid_x = {name: np.array(valid[name]) for name in feature_names}
        valid_y = np.transpose([np.array(valid[label_names])])
        test_x = {name: np.array(test[name]) for name in feature_names}
        test_y = np.transpose([np.array(test[label_names])])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.model_name == "packed_deepfm":
            num_groups = int(config.num_estimators * config.gamma)
            dnn_input_dim = int(sum(fc.embedding_dim for fc in feature_columns))
            dims_to_check = [dnn_input_dim] + list(config.hidden_units)
            bad_dims = [d for d in dims_to_check if d % num_groups != 0]
            print(
                "PackedDeepFM dim check:",
                "num_estimators",
                config.num_estimators,
                "alpha",
                config.alpha,
                "gamma",
                config.gamma,
                "num_groups",
                num_groups,
                "dnn_input_dim",
                dnn_input_dim,
                "hidden_units",
                config.hidden_units,
            )
            if bad_dims:
                raise ValueError(
                    f"PackedDeepFM requires dims divisible by num_groups={num_groups}, but got {bad_dims}"
                )

        if config.model_name == "deepfm":
            model = DeepFM(
                linear_feature_columns=feature_columns,
                dnn_feature_columns=feature_columns,
                use_fm=True,
                dnn_hidden_units=config.hidden_units,
                l2_reg_linear=config.l2_reg,
                l2_reg_embedding=config.l2_reg,
                l2_reg_dnn=config.l2_reg,
                init_std=config.init_std,
                dnn_dropout=config.dropout,
                dnn_activation="relu",
                dnn_use_bn=False,
                task="binary",
                device=device,
            )
        elif config.model_name == "packed_deepfm":
            model = PackedDeepFM(
                linear_feature_columns=feature_columns,
                dnn_feature_columns=feature_columns,
                use_fm=True,
                dnn_hidden_units=config.hidden_units,
                l2_reg_linear=config.l2_reg,
                l2_reg_embedding=config.l2_reg,
                l2_reg_dnn=config.l2_reg,
                init_std=config.init_std,
                dnn_dropout=config.dropout,
                dnn_activation="relu",
                dnn_use_bn=False,
                num_estimators=config.num_estimators,
                alpha=config.alpha,
                gamma=config.gamma,
                task="binary",
                device=device,
            )
        else:
            raise NotImplementedError

        model.compile(
            optimizer=config.optim,
            lr=config.lr,
            loss=config.loss,
            metrics=config.metrics,
            use_tune=config.use_tune,
        )

        early_stopping = EarlyStopping(
            monitor=config.monitor,
            min_delta=config.min_delta,
            verbose=config.verbose,
            patience=config.patience,
            mode=config.mode,
            restore_best_weights=config.restore_best_weights,
        )

        model_checkpoint = ModelCheckpoint(
            filepath=run_basepath,
            monitor=config.monitor,
            verbose=config.verbose,
            save_best_only=config.save_best_only,
            save_weights_only=config.save_weights_only,
            mode=config.mode,
            save_freq=config.save_freq,
            is_save=config.is_save,
        )

        class CUDAMemoryMonitor:
            def __init__(self, enabled: bool):
                self.enabled = enabled
                self.model = None

            def set_model(self, model):
                self.model = model

            def on_train_begin(self):
                if not self.enabled:
                    return
                torch.cuda.reset_peak_memory_stats()

            def on_epoch_begin(self, epoch):
                if not self.enabled:
                    return
                torch.cuda.reset_peak_memory_stats()

            def on_epoch_end(self, epoch, logs):
                if not self.enabled:
                    return
                allocated = int(torch.cuda.max_memory_allocated() / (1024**2))
                reserved = int(torch.cuda.max_memory_reserved() / (1024**2))
                print(
                    "cuda_mem_MB",
                    "epoch",
                    int(epoch) + 1,
                    "max_allocated",
                    allocated,
                    "max_reserved",
                    reserved,
                )

            def on_train_end(self):
                return

        cuda_monitor = CUDAMemoryMonitor(enabled=torch.cuda.is_available())
        try:
            history = model.fit(
                x=train_x,
                y=train_y,
                batch_size=config.batch_size,
                epochs=config.epochs,
                verbose=config.verbose,
                initial_epoch=0,
                validation_split=0.0,
                shuffle=config.shuffle,
                callbacks=[early_stopping, model_checkpoint, cuda_monitor],
                validation_data=[valid_x, valid_y],
                test_data=[test_x, test_y],
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    allocated = int(torch.cuda.max_memory_allocated() / (1024**2))
                    reserved = int(torch.cuda.max_memory_reserved() / (1024**2))
                    print(
                        "cuda_oom",
                        "max_allocated_MB",
                        allocated,
                        "max_reserved_MB",
                        reserved,
                        "batch_size",
                        config.batch_size,
                        "num_estimators",
                        config.num_estimators,
                        "alpha",
                        config.alpha,
                        "gamma",
                        config.gamma,
                    )
            raise
        if config.model_name == "packed_deepfm":
            log_packed_uncertainty_stats(
                model,
                valid_x,
                feature_names,
                min(config.batch_size, len(valid)),
                device,
            )
        return history
    finally:
        print("run_end", datetime.datetime.now().isoformat(timespec="seconds"))
        try:
            log_f.close()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    # # valid and test dataloader
    # if isinstance(valid_x, dict):
    #     valid_x = [valid_x[feature] for feature in model.feature_index]
    # for i in range(len(valid_x)):
    #     if len(valid_x[i].shape) == 1:
    #         valid_x[i] = np.expand_dims(valid_x[i], axis=1)
    # valid_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(valid_x, axis=-1)), torch.from_numpy(valid_y))
    # valid_loader = DataLoader(dataset=valid_tensor_data, shuffle=False, batch_size=1024*16, pin_memory=False)

    # if isinstance(test_x, dict):
    #     test_x = [test_x[feature] for feature in model.feature_index]
    # for i in range(len(test_x)):
    #     if len(test_x[i].shape) == 1:
    #         test_x[i] = np.expand_dims(test_x[i], axis=1)
    # test_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(test_x, axis=-1)), torch.from_numpy(test_y))
    # test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=1024*16, pin_memory=False)

    # # evaluate function
    # def evaluate(test_y_pred_calib, test_y, index):
    #     test_auc = get_auc(test_y, test_y_pred_calib)
    #     test_gauc = get_gauc(test_y, test_y_pred_calib, np.squeeze(test_x[index]))
    #     test_logloss = get_logloss(test_y, test_y_pred_calib)
    #     test_pcoc = get_pcoc(test_y, test_y_pred_calib)
    #     test_ece = get_ece(test_y, test_y_pred_calib, 100)
    #     test_fece = get_fece(test_y, test_y_pred_calib, np.squeeze(test_x[index]), 1)
    #     fece_list = get_mfece(test_y, test_y_pred_calib, test_x, 1)
    #     test_mfece = np.mean(fece_list)
    #     test_rce = get_rce(test_y, test_y_pred_calib, 100)
    #     test_frce = get_frce(test_y, test_y_pred_calib, np.squeeze(test_x[index]), 1)
    #     rce_list = get_mfrce(test_y, test_y_pred_calib, test_x, 1)
    #     test_mfrce = np.mean(rce_list)

    #     log = f"test_auc = {test_auc:.6f}, test_gauc = {test_gauc:.6f}, test_logloss = {test_logloss:.6f}, test_pcoc = {test_pcoc:.6f}, test_ece = {test_ece:.6f}, test_fece = {test_fece:.6f}, test_mfece = {test_mfece:.6f}, test_rce = {test_rce:.6f}, test_frce = {test_frce:.6f}, test_mfrce = {test_mfrce:.6f}"
    #     fece_log = f"multi-field fece list: {fece_list}"
    #     rce_log = f"multi-field rce list: {rce_list}"
    #     print(log)
    #     print(fece_log)
    #     print(rce_log)

    # test_y = []
    # test_y_pred = []
    # with torch.no_grad():
    #     for step, (x_test, y_test) in tqdm(enumerate(test_loader)):
    #         x = x_test.to(device).float()
    #         y = y_test.to(device).float()
    #         test_y.append(y.cpu().data.numpy())
    #         y_pred = model(x)
    #         test_y_pred.append(y_pred.cpu().data.numpy())
    # test_y = np.concatenate(test_y).astype("float32").flatten()
    # test_y_pred = np.concatenate(test_y_pred).astype("float32").flatten()

    # evaluate(test_y_pred, test_y, config.field_index)


if __name__ == "__main__":

    ##########################################
    use_tune = 0  # use ray.tune or not
    ##########################################

    if use_tune:
        if tune is None:
            raise ModuleNotFoundError("ray is required when use_tune=1, but it is not installed.")
        config_update = {
            "lr": tune.grid_search([1e-4, 5e-4, 1e-3]),  # 1e-4, 5e-4, 1e-3
            "l2_reg": tune.grid_search([0.0, 1e-6, 1e-5, 1e-4]),  # 0., 1e-6, 1e-5, 1e-4
        }

        analysis = tune.run(
            run_or_experiment=trial,
            config=config_update,
            resources_per_trial={"cpu": 1, "gpu": 1},
            local_dir=os.path.join(root, "ray"),
            name="",
            resume="AUTO",
        )

        metric = "val_auc"

        best_trial = analysis.get_best_trial(  # best trial is reported after stopping, so 'last' is 'best'
            metric=metric,
            mode="max",
            scope="last",
        )
        print("Best config:", best_trial.config)
        print("Best result:", best_trial.last_result)

    else:
        config_update_avazu = {
            "data_name": "avazu",
            "model_name": "packed_deepfm",
            "num_estimators": 8,
            "alpha": 1.0,
            "gamma": 1,
            "embedding_dim": 16,
            "hidden_units": [512, 256, 128, 64],
            "dropout": 0.1,
            "init_std": 1e-4,
            "l2_reg": 1e-5,
            "optim": "adam",
            "lr": 5e-4,
            "batch_size": 1024 * 16,
            "epochs": 20,
            "patience": 3,
            "shuffle": False,
            "seed": 1024,
            "use_tune": False,
            "is_save": True,
            "verbose": 2,
        }

        config_update_avazu_big = {
            "data_name": "avazu",
            "model_name": "packed_deepfm",
            "num_estimators": 16,
            "alpha": 1.0,
            "gamma": 1,
            "embedding_dim": 16,
            "hidden_units": [512, 256, 128, 64],
            "dropout": 0.1,
            "init_std": 1e-4,
            "l2_reg": 1e-5,
            "optim": "adam",
            "lr": 5e-4,
            "batch_size": 1024 * 32,
            "epochs": 20,
            "patience": 3,
            "shuffle": False,
            "seed": 1024,
            "use_tune": False,
            "is_save": True,
            "verbose": 2,
        }

        config_update_aliccp = {
            "data_name": "aliccp",
            "model_name": "packed_deepfm",
            "num_estimators": 16,
            "alpha": 1.0,
            "gamma": 1,
            "embedding_dim": 16,
            "hidden_units": [512, 256, 128, 64],
            "dropout": 0.1,
            "init_std": 1e-4,
            "l2_reg": 1e-5,
            "optim": "adam",
            "lr": 5e-4,
            "batch_size": 1024 * 32,
            "epochs": 20,
            "patience": 3,
            "shuffle": False,
            "seed": 1024,
            "use_tune": False,
            "is_save": True,
            "verbose": 2,
        }

        config_update_criteo = {
            "data_name": "criteo",
            "model_name": "packed_deepfm",
            "num_estimators": 16,
            "alpha": 1.0,
            "gamma": 1,
            "embedding_dim": 16,
            "hidden_units": [512, 256, 128, 64],
            "dropout": 0.1,
            "init_std": 1e-4,
            "l2_reg": 1e-5,
            "optim": "adam",
            "lr": 5e-4,
            "batch_size": 1024 * 32,
            "epochs": 20,
            "patience": 3,
            "shuffle": False,
            "seed": 1024,
            "use_tune": False,
            "is_save": True,
            "verbose": 2,
        }

        # config_update = config_update_avazu_big
        # config_update = config_update_avazu
        # config_update = config_update_aliccp
        config_update = config_update_criteo
        trial(config_update=config_update)
