import torch
import torch.nn as nn

from models.basemodel import BaseModel
from models.inputs import combined_dnn_input
from models.layers.interaction import FM
from models.layers.core import DNN
from models.layers.activation import activation_layer


class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        use_fm=True,
        dnn_hidden_units=(256, 128),
        l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0.0,
        init_std=0.0001,
        dnn_dropout=0.0,
        dnn_activation="relu",
        dnn_use_bn=False,
        task="binary",
        device="cpu",
        gpus=None,
    ):

        super(DeepFM, self).__init__(
            linear_feature_columns,
            dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            task=task,
            device=device,
            gpus=gpus,
        )

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(
                self.compute_input_dim(dnn_feature_columns),
                dnn_hidden_units,
                activation=dnn_activation,
                l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn,
                init_std=init_std,
                device=device,
            )
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)
        return y_pred


class PackedDNN(nn.Module):
    def __init__(
        self,
        inputs_dim,
        hidden_units,
        num_estimators,
        alpha,
        gamma,
        activation="relu",
        dropout_rate=0.0,
        use_bn=False,
        init_std=0.0001,
        dice_dim=3,
        seed=1024,
        device="cpu",
    ):
        super(PackedDNN, self).__init__()
        from torch_uncertainty.layers import PackedLinear

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [
                PackedLinear(
                    hidden_units[i],
                    hidden_units[i + 1],
                    alpha=alpha,
                    num_estimators=num_estimators,
                    gamma=gamma,
                    first=i == 0,
                    last=False,
                )
                for i in range(len(hidden_units) - 1)
            ]
        )

        if self.use_bn:
            self.bn = nn.ModuleList(
                [
                    nn.BatchNorm1d(int(hidden_units[i + 1] * alpha))
                    for i in range(len(hidden_units) - 1)
                ]
            )

        self.activation_layers = nn.ModuleList(
            [
                activation_layer(
                    activation, int(hidden_units[i + 1] * alpha), dice_dim
                )
                for i in range(len(hidden_units) - 1)
            ]
        )

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class PackedDeepFM(BaseModel):
    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        use_fm=True,
        dnn_hidden_units=(256, 128),
        l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0.0,
        init_std=0.0001,
        dnn_dropout=0.0,
        dnn_activation="relu",
        dnn_use_bn=False,
        num_estimators=4,
        alpha=1.0,
        gamma=1,
        task="binary",
        device="cpu",
        gpus=None,
    ):
        super(PackedDeepFM, self).__init__(
            linear_feature_columns,
            dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            task=task,
            device=device,
            gpus=gpus,
        )

        from torch_uncertainty.layers import PackedLinear

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        self.num_estimators = num_estimators
        self.alpha = alpha
        self.gamma = gamma
        self.logits_mean = None
        self.sigma2_epistemic = None

        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = PackedDNN(
                self.compute_input_dim(dnn_feature_columns),
                dnn_hidden_units,
                num_estimators=num_estimators,
                alpha=alpha,
                gamma=gamma,
                activation=dnn_activation,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn,
                init_std=init_std,
                device=device,
            )
            self.dnn_linear = PackedLinear(
                dnn_hidden_units[-1],
                1,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
                last=True,
            ).to(device)

            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit_mb = self.dnn_linear(dnn_output)

            base_logit_mb = (
                logit.unsqueeze(0)
                .expand(self.num_estimators, logit.shape[0], 1)
                .reshape(-1, 1)
            )
            logits_mb = base_logit_mb + dnn_logit_mb
            logits_m_b_1 = logits_mb.reshape(self.num_estimators, logit.shape[0], 1)
            logits_mean = logits_m_b_1.mean(dim=0)
            sigma2_epistemic = logits_m_b_1.var(dim=0, unbiased=False)

            self.logits_mean = logits_mean
            self.sigma2_epistemic = sigma2_epistemic
            y_pred = self.out(logits_mean)
        else:
            self.logits_mean = logit
            self.sigma2_epistemic = torch.zeros_like(logit)
            y_pred = self.out(logit)
        return y_pred
