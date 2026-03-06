import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.data import Data
from torch_geometric.utils import scatter


class WaveData(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.data = xr.open_dataset(data_path)
        self.u = self.data["u"].values # (time, N)

        self.mean = self.u.mean()
        self.std = max(self.u.std(), 1e-8)

        edge_index = self.data["edge_index"]  # expected shape (2, E)
        edge_index = np.asarray(edge_index, dtype=np.int64)
        if edge_index.shape[0] != 2:
            raise ValueError(f"edges must have shape (2,E). Got {edge_index.shape}")
 
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

        self.T = int(self.data.sizes["time"])
        if self.T < 2:
            raise ValueError("Need at least 2 time steps.")

    def __len__(self):
        return self.T - 1

    def __getitem__(self, idx: int):
        x = self.data["u"].isel(time=idx).values      # (N,)
        y = self.data["u"].isel(time=idx + 1).values  # (N,)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)     # (N,1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)   # (N,1)

        # PyG graph object
        return Data(x=x, y=y, edge_index=self.edge_index)


class GNNModel(pl.LightningModule):
    def __init__(self, phi: nn.Module, psi: nn.Module, aggregate: str = "mean", lr: float = 1e-3):
        super().__init__()
        self.phi = phi
        self.psi = psi
        self.aggregate = aggregate
        self.lr = lr
        self.train_losses = []

    def forward(self, data: Data):
        """
        data.x: (N_total, Fin)
        data.edge_index: (2, E_total)
        data is a PyG Batch during training.
        """
        x = data.x
        edge_index = data.edge_index
        src, dst = edge_index[0], edge_index[1]  # (E,), (E,)

        # messages on edges: m_e = phi([x_src, x_dst])
        x_src = x[src]
        x_dst = x[dst]
        m_in = torch.cat([x_src, x_dst], dim=-1)   # (E, 2*Fin)
        messages = self.phi(m_in)                  # (E, Fm)

        # aggregate messages per destination node
        # scatter supports reduce="sum" or "mean"
        if self.aggregate == "sum":
            agg = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
        elif self.aggregate == "mean":
            agg = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="mean")
        elif self.aggregate == "max":
            agg = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="max")
        else:
            raise ValueError(f"Unknown aggregate='{self.aggregate}'")

        # update: x_next = psi([x, agg])
        u_in = torch.cat([x, agg], dim=-1)   # (N_total, Fin+Fm)
        x_next = x + self.psi(u_in)             # (N_total, Fin_out)
        return x_next 

    def training_step(self, batch: Data, batch_idx: int):
        pred = self(batch)
        loss = F.smooth_l1_loss(pred,batch.y,beta=0.01) #F.mse_loss(pred, batch.y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs)

        self.train_losses.append(loss.detach().cpu().item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

