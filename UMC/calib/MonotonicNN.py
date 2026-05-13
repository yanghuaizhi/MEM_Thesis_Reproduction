import torch
import torch.nn as nn
import torch.nn.functional as F
from calib.ParallelNeuralIntegral import ParallelNeuralIntegral
from utils.inputs import *


class IntegrandNN(nn.Module):
    def __init__(self, in_dim, hidden_layers, device):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_dim] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())

        self.net = nn.Sequential(*self.net)
        self.net = self.net.to(device)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.0


class UMC(nn.Module):
    def __init__(
        self,
        hidden_layers,
        feature_columns,
        feature_index,
        device,
        nb_steps=50,
        rescaling=True,
    ):
        super(UMC, self).__init__()
        in_dim = 1 + compute_input_dim(feature_columns)
        self.integrand = IntegrandNN(in_dim, hidden_layers, device=device)
        self.rescaling = rescaling

        b = torch.zeros(1).to(device)
        self.bias = nn.Parameter(b)

        in_dim = compute_input_dim(feature_columns)

        self.net = []
        hs = [in_dim] + [200, 200] + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net).to(device)

        self.device = device
        self.nb_steps = nb_steps
        self.feature_columns = feature_columns
        self.feature_index = feature_index

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.trunc_normal_(m.bias, 0.0, 1e-4)

    def forward(self, x, y, embedding_dict):

        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.feature_columns, embedding_dict, self.feature_index, self.device
        )
        y0 = torch.zeros_like(y).to(self.device)
        h_out = torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1)
        h_in = h_out

        # stop gradient
        h_out = h_out.detach()
        h_in = h_in.detach()
        y = y.detach()

        offset = self.bias
        weight = torch.exp(self.net(h_out)[:, [0]])
        bias = self.net(h_out)[:, [1]]

        result = (
            ParallelNeuralIntegral.apply(
                y0,
                y,
                self.integrand,
                flatten(self.integrand.parameters()),
                h_in,
                self.nb_steps,
            )
            + offset
        )
        if self.rescaling:
            result = weight * result + bias

        return result


class UMNN(nn.Module):
    def __init__(self, hidden_layers, device, nb_steps=50):
        super(UMNN, self).__init__()
        in_dim = 2
        self.integrand = IntegrandNN(in_dim, hidden_layers, device=device)

        b = torch.zeros(1).to(device)
        self.bias = nn.Parameter(b)

        self.device = device
        self.nb_steps = nb_steps

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.trunc_normal_(m.bias, 0.0, 1e-4)

    def forward(self, y):
        y0 = torch.zeros_like(y).to(self.device)
        h = torch.zeros_like(y).to(self.device)

        # stop gradient
        h = h.detach()
        y = y.detach()

        offset = self.bias
        result = (
            ParallelNeuralIntegral.apply(
                y0,
                y,
                self.integrand,
                flatten(self.integrand.parameters()),
                h,
                self.nb_steps,
            )
            + offset
        )
        return result


class UAMCM(nn.Module):
    def __init__(
        self,
        hidden_layers,
        feature_columns,
        feature_index,
        device,
        nb_steps=50,
        rescaling=True,
        u_min=-20.0,
        u_max=20.0,
        u_in_rescaling=True,
    ):
        super(UAMCM, self).__init__()
        in_dim = 1 + compute_input_dim(feature_columns) + 1
        self.integrand = IntegrandNN(in_dim, hidden_layers, device=device)
        self.rescaling = rescaling
        self.u_in_rescaling = u_in_rescaling

        b = torch.zeros(1).to(device)
        self.bias = nn.Parameter(b)

        if u_in_rescaling:
            rescale_in_dim = compute_input_dim(feature_columns) + 1  # ctx + u
        else:
            rescale_in_dim = compute_input_dim(feature_columns)      # ctx only

        self.net = []
        hs = [rescale_in_dim] + [200, 200] + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()
        self.net = nn.Sequential(*self.net).to(device)

        self.device = device
        self.nb_steps = nb_steps
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.trunc_normal_(m.bias, 0.0, 1e-4)

    def forward(self, x, y, u, embedding_dict):
        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.feature_columns, embedding_dict, self.feature_index, self.device
        )
        y0 = torch.zeros_like(y).to(self.device)
        h_ctx = torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1)
        u = u.to(self.device).reshape(-1, 1)
        u = torch.clamp(u, min=self.u_min, max=self.u_max)
        h_in = torch.cat([h_ctx, u], dim=1)

        h_ctx = h_ctx.detach()
        h_in = h_in.detach()
        y = y.detach()

        if self.u_in_rescaling:
            h_rs = h_in    # rescaling sees ctx + u
        else:
            h_rs = h_ctx   # rescaling sees ctx only

        offset = self.bias
        weight = torch.exp(self.net(h_rs)[:, [0]])
        bias = self.net(h_rs)[:, [1]]

        result = (
            ParallelNeuralIntegral.apply(
                y0,
                y,
                self.integrand,
                flatten(self.integrand.parameters()),
                h_in,
                self.nb_steps,
            )
            + offset
        )
        if self.rescaling:
            result = weight * result + bias
        return result


class UAMCMPhase4(nn.Module):
    def __init__(
        self,
        hidden_layers,
        feature_columns,
        feature_index,
        device,
        nb_steps=50,
        rescaling=True,
        u_min=-20.0,
        u_max=20.0,
        alpha_max=1.0,
        delta_scale_init=0.1,
    ):
        super(UAMCMPhase4, self).__init__()
        in_dim = 1 + compute_input_dim(feature_columns) + 1
        self.integrand = IntegrandNN(in_dim, hidden_layers, device=device)
        self.rescaling = rescaling

        b = torch.zeros(1).to(device)
        self.bias = nn.Parameter(b)

        in_dim = compute_input_dim(feature_columns) + 1
        self.net = []
        hs = [in_dim] + [200, 200] + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()
        self.net = nn.Sequential(*self.net).to(device)

        self.device = device
        self.nb_steps = nb_steps
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        self.alpha_max = float(alpha_max)
        self.alpha_w_raw = nn.Parameter(torch.zeros(1).to(device))
        self.alpha_b = nn.Parameter(torch.zeros(1).to(device))
        self.delta_scale = nn.Parameter(
            torch.tensor(float(delta_scale_init)).to(device)
        )
        self.alpha_value = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.trunc_normal_(m.bias, 0.0, 1e-4)

    def forward(self, x, y, u, embedding_dict):
        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.feature_columns, embedding_dict, self.feature_index, self.device
        )
        y0 = torch.zeros_like(y).to(self.device)
        h_out = torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1)
        u = u.to(self.device).reshape(-1, 1)
        u = torch.clamp(u, min=self.u_min, max=self.u_max)
        h_in = torch.cat([h_out, u], dim=1)

        h_out = h_in.detach()
        h_in = h_in.detach()
        y = y.detach()

        offset = self.bias
        weight = torch.exp(self.net(h_out)[:, [0]])
        bias = self.net(h_out)[:, [1]]

        result = (
            ParallelNeuralIntegral.apply(
                y0,
                y,
                self.integrand,
                flatten(self.integrand.parameters()),
                h_in,
                self.nb_steps,
            )
            + offset
        )
        if self.rescaling:
            result = weight * result + bias

        delta = result - y
        alpha_w = F.softplus(self.alpha_w_raw)
        alpha = self.alpha_max * torch.sigmoid(alpha_w * u + self.alpha_b)
        self.alpha_value = alpha
        return y + alpha * self.delta_scale * delta


class UASAC(nn.Module):
    """Uncertainty-Aware Stratified Adaptive Calibration.

    Routes samples to K implicit monotonic experts via uncertainty-aware routing,
    blending expert embeddings as conditioning for a shared UMNN integrator.
    Single integration pass regardless of K.
    """

    def __init__(
        self,
        hidden_layers,
        feature_columns,
        feature_index,
        device,
        nb_steps=50,
        num_experts=3,
        expert_dim=16,
        router_hidden=64,
        router_type="mlp",
        temperature=1.0,
        router_mode="full",
    ):
        super(UASAC, self).__init__()

        ctx_dim = compute_input_dim(feature_columns)
        cond_dim = ctx_dim + expert_dim

        self.integrand = IntegrandNN(1 + cond_dim, hidden_layers, device=device)

        self.bias = nn.Parameter(torch.zeros(1).to(device))

        self.expert_embeds = nn.Parameter(
            torch.randn(num_experts, expert_dim, device=device) * 0.01
        )

        self.router_mode = router_mode
        if router_mode == "u_only":
            router_in_dim = 1
        else:
            router_in_dim = 1 + ctx_dim  # "full" and "random" both use this shape

        if router_mode != "random":
            if router_type == "mlp":
                self.router = nn.Sequential(
                    nn.Linear(router_in_dim, router_hidden),
                    nn.ReLU(),
                    nn.Linear(router_hidden, num_experts),
                ).to(device)
            else:
                self.router = nn.Linear(router_in_dim, num_experts).to(device)
        else:
            self.router = None

        self.device = device
        self.nb_steps = nb_steps
        self.temperature = temperature
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.num_experts = num_experts

        self.router_weights = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.trunc_normal_(m.bias, 0.0, 1e-4)

    def forward(self, x, y, u, embedding_dict):
        sparse_embedding_list, _ = input_from_feature_columns(
            x, self.feature_columns, embedding_dict, self.feature_index, self.device
        )
        h_ctx = torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1)

        u = u.to(self.device).reshape(-1, 1)
        if self.router_mode == "random":
            w = F.softmax(torch.randn(u.size(0), self.num_experts, device=self.device), dim=1)
        elif self.router_mode == "u_only":
            logits_router = self.router(u)
            w = F.softmax(logits_router / self.temperature, dim=1)
        else:
            router_input = torch.cat([u, h_ctx.detach()], dim=1)
            logits_router = self.router(router_input)
            w = F.softmax(logits_router / self.temperature, dim=1)
        self.router_weights = w.detach()

        e_blend = torch.matmul(w, self.expert_embeds)

        h_in = torch.cat([h_ctx.detach(), e_blend], dim=1)

        y0 = torch.zeros_like(y).to(self.device)
        y_det = y.detach()

        result = (
            ParallelNeuralIntegral.apply(
                y0,
                y_det,
                self.integrand,
                flatten(self.integrand.parameters()),
                h_in,
                self.nb_steps,
            )
            + self.bias
        )

        return result


class UASAC_R(UASAC):
    """UASAC with rescaling (weight * integral + bias)."""

    def __init__(
        self,
        hidden_layers,
        feature_columns,
        feature_index,
        device,
        nb_steps=50,
        num_experts=3,
        expert_dim=16,
        router_hidden=64,
        router_type="mlp",
        temperature=1.0,
        router_mode="full",
    ):
        super(UASAC_R, self).__init__(
            hidden_layers, feature_columns, feature_index, device,
            nb_steps, num_experts, expert_dim,
            router_hidden, router_type, temperature,
            router_mode,
        )
        ctx_dim = compute_input_dim(feature_columns)
        cond_dim = ctx_dim + expert_dim
        self.rescale_net = nn.Sequential(
            nn.Linear(cond_dim, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 2),
        ).to(device)
        for m in self.rescale_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.trunc_normal_(m.bias, 0.0, 1e-4)

    def forward(self, x, y, u, embedding_dict):
        sparse_embedding_list, _ = input_from_feature_columns(
            x, self.feature_columns, embedding_dict, self.feature_index, self.device
        )
        h_ctx = torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1)

        u = u.to(self.device).reshape(-1, 1)
        if self.router_mode == "random":
            w = F.softmax(torch.randn(u.size(0), self.num_experts, device=self.device), dim=1)
        elif self.router_mode == "u_only":
            logits_router = self.router(u)
            w = F.softmax(logits_router / self.temperature, dim=1)
        else:
            router_input = torch.cat([u, h_ctx.detach()], dim=1)
            logits_router = self.router(router_input)
            w = F.softmax(logits_router / self.temperature, dim=1)
        self.router_weights = w.detach()

        e_blend = torch.matmul(w, self.expert_embeds)
        h_in = torch.cat([h_ctx.detach(), e_blend], dim=1)

        y0 = torch.zeros_like(y).to(self.device)
        y_det = y.detach()

        result = (
            ParallelNeuralIntegral.apply(
                y0, y_det, self.integrand,
                flatten(self.integrand.parameters()),
                h_in, self.nb_steps,
            )
            + self.bias
        )

        rs_out = self.rescale_net(h_in.detach())
        weight = torch.exp(rs_out[:, [0]])
        bias_rs = rs_out[:, [1]]
        result = weight * result + bias_rs

        return result
