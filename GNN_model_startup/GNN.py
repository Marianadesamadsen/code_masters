import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.data import Data
from torch_geometric.utils import scatter

 
class WaveData(torch.utils.data.Dataset):
    def __init__(self, ds, data_path = None):
        super().__init__()
        
        if data_path is not None:
            self.data = xr.open_dataset(data_path)
        else: 
            self.data = ds

        self.u = self.data["u"].values 

        self.mean = self.u.mean()
        self.std = max(self.u.std(), 1e-8)

        edge_index = self.data["edge_index"]  
        edge_index = np.asarray(edge_index, dtype=np.int64)

        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

        self.T = int(self.data.sizes["time"])

    def __len__(self):
        return self.T - 1

    def __getitem__(self, idx: int):
        x = self.data["u"].isel(time=idx).values      
        y = self.data["u"].isel(time=idx + 1).values 

        # Normalize data
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)   
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  

        return Data(x=x, y=y, edge_index=self.edge_index)


class GNNModel(pl.LightningModule):
    def __init__(self, phi: nn.Module, psi: nn.Module, aggregate: str = "mean", lr: float = 1e-3):
        super().__init__()
        self.phi = phi 
        self.psi = psi
        self.aggregate = aggregate
        self.lr = lr

    def forward(self, data: Data):

        x = data.x
        edge_index = data.edge_index
        src, dst = edge_index[0], edge_index[1]  # First nodes are the sources, second nodes are destinations

        x_src = x[src]
        x_dst = x[dst]
        m_in = torch.cat([x_src, x_dst], dim=-1)  
        messages = self.phi(m_in)                 

        if self.aggregate == "sum":
            agg = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
        elif self.aggregate == "mean":
            agg = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="mean")
        elif self.aggregate == "max":
            agg = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="max")
        else:
            raise ValueError(f"Unknown aggregate='{self.aggregate}'")

        u_in = torch.cat([x, agg], dim=-1)   
        x_next = x + self.psi(u_in)       
        return x_next 

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class MetricsLogger(pl.Callback):

    # https://www.geeksforgeeks.org/deep-learning/extracting-loss-and-accuracy-from-pytorch-lightning-logger/
    def __init__(self):
        super().__init__()
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", None)

        if train_loss is not None:
            train_loss = train_loss.item()
            self.train_losses.append(train_loss)
            print(f"Epoch {trainer.current_epoch}: train loss = {train_loss:.4f}")
