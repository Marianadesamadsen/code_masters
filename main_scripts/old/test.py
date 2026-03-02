# gnn_ode_with_plots.py
# Train a message-passing GNN to learn one-step dynamics of dx/dt = λx
# Compare:
#   (A) direct:   x_{t+1} = model(x_t)
#   (B) residual: x_{t+1} = x_t + model_delta(x_t)
# Plot one-step loss curves + rollout error curves.

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -----------------
# Reproducibility
# -----------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# -----------------
# Ring graph (directed edges: neighbors -> node)
# edge_index: [2, E] with src, dst
# -----------------
def ring_edge_index(n_nodes: int) -> torch.Tensor:
    src, dst = [], []
    for i in range(n_nodes):
        left = (i - 1) % n_nodes
        right = (i + 1) % n_nodes
        src += [left, right]
        dst += [i, i]
    return torch.tensor([src, dst], dtype=torch.long)


# -----------------
# Dataset: exact one-step mapping of linear ODE
# x_{t+dt} = exp(lam*dt) * x_t
# Each sample is a whole graph state: [N, F]
# -----------------
class LinearODEGraphDataset(Dataset):
    def __init__(self, n_samples: int, n_nodes: int, feat_dim: int, dt: float, lam: float):
        super().__init__()
        self.mult = math.exp(lam * dt)
        x_t = torch.randn(n_samples, n_nodes, feat_dim)
        x_tp1 = self.mult * x_t
        self.x_t = x_t
        self.x_tp1 = x_tp1

    def __len__(self):
        return self.x_t.shape[0]

    def __getitem__(self, idx: int):
        return self.x_t[idx], self.x_tp1[idx]


# -----------------
# Model blocks
# -----------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MessagePassingLayer(nn.Module):
    """
    h: [B, N, H]
    message: m_{src->dst} = MLP([h_src, h_dst])
    agg: sum incoming messages per dst
    update: h <- h + MLP([h, agg])   (residual within layer)
    """
    def __init__(self, hidden_dim: int, mlp_dim: int, mp_residual: bool = True):
        super().__init__()
        self.mp_residual = mp_residual
        self.mlp_msg = MLP(2 * hidden_dim, mlp_dim, hidden_dim)
        self.mlp_upd = MLP(2 * hidden_dim, mlp_dim, hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, H = h.shape
        src, dst = edge_index  # [E], [E]

        h_src = h[:, src, :]   # [B, E, H]
        h_dst = h[:, dst, :]
        m = self.mlp_msg(torch.cat([h_src, h_dst], dim=-1))  # [B, E, H]

        agg = torch.zeros(B, N, H, device=h.device)
        agg.index_add_(1, dst, m)  # sum messages into dst nodes

        upd = self.mlp_upd(torch.cat([h, agg], dim=-1))  # [B, N, H]
        return h + upd if self.mp_residual else upd


class GNNStepper(nn.Module):
    """
    encode -> K message passing layers -> decode
    If time_residual=True: x_{t+1} = x_t + delta
    Else:                 x_{t+1} = direct_pred
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        mlp_dim: int,
        mp_residual: bool,
        time_residual: bool,
    ):
        super().__init__()
        self.time_residual = time_residual
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, mlp_dim, mp_residual=mp_residual) for _ in range(n_layers)]
        )
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_t)
        for layer in self.layers:
            h = layer(h, edge_index)
        out = self.decoder(h)  # delta or direct next state
        return x_t + out if self.time_residual else out


# -----------------
# Train + rollout
# -----------------
def train_model(model, train_loader, val_loader, edge_index, epochs=10, lr=3e-4):
    model = model.to(device)
    edge_index = edge_index.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_curve, val_curve = [], []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x_t, x_tp1 in train_loader:
            x_t = x_t.to(device)
            x_tp1 = x_tp1.to(device)

            pred = model(x_t, edge_index)
            loss = F.mse_loss(pred, x_tp1)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        vlosses = []
        with torch.no_grad():
            for x_t, x_tp1 in val_loader:
                x_t = x_t.to(device)
                x_tp1 = x_tp1.to(device)
                pred = model(x_t, edge_index)
                vlosses.append(F.mse_loss(pred, x_tp1).item())

        train_curve.append(float(np.mean(losses)))
        val_curve.append(float(np.mean(vlosses)))

        if ep in (1, 2, 5, epochs):
            print(f"Epoch {ep:02d} | train MSE={train_curve[-1]:.3e} | val MSE={val_curve[-1]:.3e}")

    return train_curve, val_curve


def rollout_rel_mse(model, x0, edge_index, mult, steps=50):
    """
    Compare rollout vs exact: x_k = mult^k * x0
    Returns relative MSE per step.
    """
    model.eval()
    x0 = x0.to(device)
    edge_index = edge_index.to(device)
    x_pred = x0.clone()

    rel = []
    with torch.no_grad():
        for k in range(1, steps + 1):
            x_pred = model(x_pred, edge_index)
            x_true = (mult ** k) * x0

            mse = F.mse_loss(x_pred, x_true).item()
            power = (x_true ** 2).mean().item() + 1e-12
            rel.append(mse / power)
    return rel


def main():
    # ---- Small defaults; increase once it works fast on your machine
    n_nodes = 32
    feat_dim = 8
    dt = 0.1
    lam = -0.5  # try +0.5 to make it harder

    n_train = 4000
    n_val = 800
    batch_size = 128
    epochs = 5

    hidden_dim = 64
    mlp_dim = 128
    n_layers = 3
    mp_residual = True

    edge_index = ring_edge_index(n_nodes)

    train_ds = LinearODEGraphDataset(n_train, n_nodes, feat_dim, dt, lam)
    val_ds = LinearODEGraphDataset(n_val, n_nodes, feat_dim, dt, lam)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Direct model
    model_direct = GNNStepper(
        in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=feat_dim,
        n_layers=n_layers, mlp_dim=mlp_dim,
        mp_residual=mp_residual, time_residual=False
    )

    # Residual time-step model
    model_resid = GNNStepper(
        in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=feat_dim,
        n_layers=n_layers, mlp_dim=mlp_dim,
        mp_residual=mp_residual, time_residual=True
    )

    print("\nTraining DIRECT model (predict x_{t+1})")
    train_d, val_d = train_model(model_direct, train_loader, val_loader, edge_index, epochs=epochs)

    print("\nTraining RESIDUAL model (predict Δx, output x+Δx)")
    train_r, val_r = train_model(model_resid, train_loader, val_loader, edge_index, epochs=epochs)

    # Rollout
    mult = math.exp(lam * dt)
    x0 = torch.randn(1, n_nodes, feat_dim)
    steps = 80
    rel_d = rollout_rel_mse(model_direct, x0, edge_index, mult, steps=steps)
    rel_r = rollout_rel_mse(model_resid,  x0, edge_index, mult, steps=steps)

    # ---- Plot & save
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    fig1 = os.path.join(out_dir, "one_step_loss.png")
    fig2 = os.path.join(out_dir, "rollout_rel_mse.png")

    plt.figure()
    plt.plot(train_d, label="train (direct)")
    plt.plot(val_d, label="val (direct)")
    plt.plot(train_r, label="train (residual)")
    plt.plot(val_r, label="val (residual)")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE (log)")
    plt.title("One-step training/validation loss")
    plt.legend()
    plt.savefig(fig1, dpi=180, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(rel_d, label="direct")
    plt.plot(rel_r, label="residual")
    plt.yscale("log")
    plt.xlabel("rollout step")
    plt.ylabel("relative MSE (log)")
    plt.title("Rollout error vs exact ODE solution")
    plt.legend()
    plt.savefig(fig2, dpi=180, bbox_inches="tight")
    plt.show()

    print("\nSaved plots:")
    print(" ", fig1)
    print(" ", fig2)


if __name__ == "__main__":
    main()


