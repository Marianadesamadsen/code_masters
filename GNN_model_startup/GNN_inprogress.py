from torch.utils.data import Dataset as Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import xarray as xr
import numpy as np

def tri_to_edge_index(tri: np.ndarray, make_undirected: bool = True) -> np.ndarray:
    """
    tri: (F, 3) triangle indices
    returns edge_index: (2, E)
    """
    tri = np.asarray(tri, dtype=np.int64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

    edges = np.concatenate(
        [
            np.stack([a, b], axis=0),
            np.stack([b, c], axis=0),
            np.stack([c, a], axis=0),
        ],
        axis=1,
    )  # (2, 3F)

    if make_undirected:
        edges_rev = edges[[1, 0], :]
        edges = np.concatenate([edges, edges_rev], axis=1)

    edges_t = np.unique(edges.T, axis=0)  # (E,2) unique
    return edges_t.T  # (2,E)


class WaveData(Dataset):
    def __init__(self, path: str, tri_path: str):
        super().__init__()
        self.data = xr.open_dataset(path)
        triangles = np.load(tri_path)  # (F,3)
        self.edge_index = self.edge_index = torch.tensor(tri_to_edge_index(triangles), dtype=torch.long)  # (2,E)

    def __len__(self):
        return self.data.sizes["time"] - 1
    
    def __getitem__(self, idx):

        xt = self.data["u"].isel(time=idx).values
        xtp1 = self.data["u"].isel(time=idx + 1).values  

        xt = torch.tensor(xt, dtype=torch.float32).unsqueeze(-1)
        xtp1 = torch.tensor(xtp1, dtype=torch.float32).unsqueeze(-1)   
        return xt,xtp1,self.edge_index

class GNNModel(pl.LightningModule):
    def __init__(self,phi,psi,aggregate,lr):
        super().__init__()
        self.phi = phi
        self.psi = psi
        self.aggregate = aggregate
        self.lr = lr
        self.train_losses = []
    
    def forward(self,xt,edge_index):

        messages = self.message_passing(edge_index,xt)
        aggregated_messages = self.aggregation(messages)
        xtp1 = self.update(aggregated_messages)

        return xtp1

    def training_step(self,batch,batch_idx): 
        xt,xtp1,edge_index = batch

        xtp1_hat = self.forward(xt,edge_index)
        loss = F.mse_loss(xtp1_hat,xtp1)
        self.train_losses.append(loss)

        return loss
    
    def configure_optimizers(self):
        
        return torch.optim.Adam(self.parameters(),lr = self.lr)
    
    def aggregation(self,messsages):
        
        if self.aggregate == "mean":
            ag = torch.mean(messsages)
        elif self.aggregate == "sum":
            ag = torch.sum(messsages)
        elif self.aggregate == "max":
            ag = torch.max(messsages)
        
        return ag

    def update(self, data, aggregated_messages):
        u_in = torch.cat([data, aggregated_messages], dim=-1)  # (N, Fin+Fm)
        return self.psi(u_in)                                   # (N, Fin)

    def message_passing(self,edge_index,data):
        # edge_index: (2,E)
        j,i = edge_index[0], edge_index[1]  # (E,), (E,)
        x_j = data[j]  # (E, Fin)
        x_i = data[i]  # (E, Fin)

        m_in = torch.cat([x_j, x_i], dim=-1)  # (E, 2*Fin)
        messages = self.phi(m_in)              # (E, Fm)
        return messages
    

if __name__ == "__main":

    dataset = WaveData(path="wave_sphere_data_test.nc", tri_path="wave_sphere_data_test_tri.npy")

    # Batch properly with collate_fn (so batch_size>1 works)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_graphs,
    )





