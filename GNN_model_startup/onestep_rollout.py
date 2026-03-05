import xarray as xr
import numpy as np
import torch
from torch_geometric.data import Data

def one_step_rollout(model, ds, steps=None, device="cpu"):

    model.eval()
    model.to(device)

    u_true = ds["u"].values
    if u_true.ndim != 2:
        raise ValueError("rollout_model expects ds['u'] with dims (time, grid_index)")

    T, N = u_true.shape
    if steps is None:
        steps = T - 1
    steps = min(steps, T - 1)

    # get edge_index
    edge_index = torch.tensor(ds["edge_index"].values, dtype=torch.long, device=device)

    # initial condition
    x = torch.tensor(u_true[0], dtype=torch.float32, device=device).unsqueeze(-1)  # (N,1)

    # First prediction is just the initial condition
    preds = [x.squeeze(-1).detach().cpu().numpy()] 

    with torch.no_grad():
        for step in range(steps):
            # Autoregressive step: predict next state from current state
            x = torch.tensor(u_true[step], dtype=torch.float32, device=device).unsqueeze(-1)  # (N,1)
            graph = Data(x=x, edge_index=edge_index)
            x = model(graph)  # (N,1)
            preds.append(x.squeeze(-1).detach().cpu().numpy())

    u_pred = np.stack(preds, axis=0)  # (steps+1, N) 

    # build dataset for plotting (reuse mesh vars if present) 
    out = xr.Dataset(
        data_vars={"u": (("time", "grid_index"), u_pred)},
        coords={
            "time": ds["time"].values[: steps + 1],
            "grid_index": np.arange(N),
        },
        attrs=dict(ds.attrs),
    )

    # carry over plotting helpers if available 
    for v in ["P", "tri", "x", "y", "z", "x_static", "y_static", "z_static"]:
        if v in ds:
            out[v] = ds[v]

    return out