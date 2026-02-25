from torch.utils.data import Dataset as TorchDataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

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


class WaveDataset(TorchDataset):
    def __init__(self, path: str, tri_path: str):
        self.data = xr.open_dataset(path)

        tri = np.load(tri_path)  # (F,3)
        self.edge_index = torch.tensor(tri_to_edge_index(tri), dtype=torch.long)  # (2,E)

        self.T = int(self.data.sizes["time"])
        if self.T < 2:
            raise ValueError("Need at least 2 time steps for (t -> t+1) training pairs.")

    def __len__(self):
        return self.T - 1

    def __getitem__(self, idx: int):
        # u: (time, grid_index) -> select time -> (N,)
        x = self.data["u"].isel(time=idx).values
        y = self.data["u"].isel(time=idx + 1).values

        # to tensors (N,1)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return {"input": x, "target": y, "edge_index": self.edge_index}


def collate_graphs(batch):
    """
    Batch graphs by concatenating nodes and offsetting edge indices.
    Each item:
      input: (N,Fin), target: (N,Fout), edge_index: (2,E)
    Output:
      input: (B*N,Fin), target: (B*N,Fout), edge_index: (2,B*E)
    """
    xs = [b["input"] for b in batch]
    ys = [b["target"] for b in batch]
    edge_index = batch[0]["edge_index"]

    N = xs[0].shape[0]
    B = len(batch)

    x = torch.cat(xs, dim=0)  # (B*N, Fin)
    y = torch.cat(ys, dim=0)  # (B*N, Fout)

    eis = []
    for i in range(B):
        offset = i * N
        eis.append(edge_index + offset)
    edge_index_batched = torch.cat(eis, dim=1)  # (2, B*E)

    return {"input": x, "target": y, "edge_index": edge_index_batched}


class GNN(pl.LightningModule):
    def __init__(self, layers: nn.Module, aggr: str, upd: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.layers = layers
        self.aggr = aggr
        self.upd = upd
        self.lr = lr
        self.train_losses = []

    def forward(self, data: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        messages = self.message_passing(edge_index, data)
        aggregated_messages = self.aggregation(messages, edge_index, num_nodes=data.size(0))
        out = self.update(data, aggregated_messages)
        return out

    def training_step(self, batch, batch_idx):
        xt = batch["input"]
        xtp1 = batch["target"]
        edge_index = batch["edge_index"]

        pred = self.forward(xt, edge_index)
        loss = F.mse_loss(pred, xtp1)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.train_losses.append(loss.detach().cpu().item())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def message_passing(self, edge_index: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        # edge_index: (2,E)
        src, dst = edge_index[0], edge_index[1]  # (E,), (E,)
        x_src = data[src]  # (E, Fin)
        x_dst = data[dst]  # (E, Fin)

        m_in = torch.cat([x_src, x_dst], dim=-1)  # (E, 2*Fin)
        messages = self.layers(m_in)              # (E, Fm)
        return messages

    def aggregation(self, messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # aggregate incoming messages for each dst node
        dst = edge_index[1]  # (E,)
        Fm = messages.size(-1)

        agg = torch.zeros((num_nodes, Fm), device=messages.device, dtype=messages.dtype)
        agg.index_add_(0, dst, messages)

        if self.aggr == "mean":
            deg = torch.zeros((num_nodes,), device=messages.device, dtype=messages.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=messages.dtype))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
        elif self.aggr == "sum":
            pass
        else:
            raise ValueError(f"Unsupported aggr='{self.aggr}'. Use 'mean' or 'sum'.")

        return agg

    def update(self, data: torch.Tensor, aggregated_messages: torch.Tensor) -> torch.Tensor:
        u_in = torch.cat([data, aggregated_messages], dim=-1)  # (N, Fin+Fm)
        return self.upd(u_in)                                   # (N, Fin)


def main():
    dataset = WaveDataset(path="wave_sphere_data_test.nc", tri_path="wave_sphere_data_test_tri.npy")

    # Batch properly with collate_fn (so batch_size>1 works)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_graphs,
    )

    Fin = 1
    Fm = 64 

    layers = nn.Sequential(
        nn.Linear(2 * Fin, Fm),
        nn.ReLU(),
        nn.Linear(Fm, Fm),
    ) 

    upd = nn.Sequential(
        nn.Linear(Fin + Fm, Fm),
        nn.ReLU(),
        nn.Linear(Fm, Fin),
    )

    model = GNN(layers=layers, aggr="mean", upd=upd, lr=1e-3) 

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, dataloader)

    import matplotlib.pyplot as plt

    # ---- Plot loss ----
    plt.figure()
    plt.plot(model.train_losses)
    plt.xlabel("Training step")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.tight_layout()

    # ---- Make predictions on ONE graph (first element of batch) ----
    batch = next(iter(dataloader))

    xt = batch["input"]
    xtp1 = batch["target"]
    edge_index = batch["edge_index"]

    model.eval()
    with torch.no_grad():
        pred = model(xt, edge_index)

    # Convert to numpy
    true = xtp1.detach().cpu().numpy().reshape(-1)
    pred = pred.detach().cpu().numpy().reshape(-1)

    # Sphere coordinates (N,)
    ds = xr.open_dataset("wave_sphere_data_test.nc")
    x = ds["x"].values
    y = ds["y"].values
    z = ds["z"].values
    N = x.shape[0]

    # Because batch_size=32, true/pred have length B*N. Plot only first graph:
    true0 = true[:N]
    pred0 = pred[:N]

    # ---- 3D scatter plots ----
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    p1 = ax1.scatter(x, y, z, c=true0, cmap="viridis")
    ax1.set_title("True u(t+1) (graph 0)")
    fig.colorbar(p1, ax=ax1, shrink=0.7)

    ax2 = fig.add_subplot(122, projection="3d")
    p2 = ax2.scatter(x, y, z, c=pred0, cmap="viridis")
    ax2.set_title("Predicted u(t+1) (graph 0)")
    fig.colorbar(p2, ax=ax2, shrink=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



