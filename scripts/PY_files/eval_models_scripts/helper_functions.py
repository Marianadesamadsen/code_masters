import os 
import re
from matplotlib import colors
import torch
import numpy as np
import xarray as xr


def all_raw_files(raw_dir):
    pred_files = sorted(
        [f for f in os.listdir(raw_dir) if re.match(r"pred_batch_\d+\.pt", f)],
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )
    target_files = sorted(
        [f for f in os.listdir(raw_dir) if re.match(r"target_batch_\d+\.pt", f)],
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )
    time_files = sorted(
        [f for f in os.listdir(raw_dir) if re.match(r"time_batch_\d+\.pt", f)],
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )

    return pred_files, target_files, time_files

def concat_all_batches(raw_dir, pred_files, target_files, time_files):
    all_pred = []
    all_target = []
    all_time = []

    for pf, qf, tf in zip(pred_files, target_files, time_files):
        pred = torch.load(os.path.join(raw_dir, pf)).detach().cpu()
        target = torch.load(os.path.join(raw_dir, qf)).detach().cpu()
        times = torch.load(os.path.join(raw_dir, tf)).detach().cpu()

        all_pred.append(pred)
        all_target.append(target)
        all_time.append(times)

    pred_all = torch.cat(all_pred, dim=0)      # (n_rollouts, T, node, feature)
    target_all = torch.cat(all_target, dim=0)  # (n_rollouts, T, node, feature)
    time_all = torch.cat(all_time, dim=0)      # (n_rollouts, T)

    return pred_all, target_all, time_all


def setup_simple_xarray(u, time, P, tri, R=1):
    ds = xr.Dataset(
    data_vars={
        "u": (("time", "node"), u),
        "P": (("node", "coord"), P),
        "tri": (("face", "vertex"), tri),
    },
    coords={
        "time": time,
        "node": np.arange(P.shape[0]),
    },
    attrs={"R": R},
    )

    return ds    

def compute_errors(u_pred, u_true, axis=1):
    if torch.is_tensor(u_pred):
        u_pred = u_pred.detach().cpu().numpy()
    if torch.is_tensor(u_true):
        u_true = u_true.detach().cpu().numpy()

    err = u_pred - u_true
    rmse = np.sqrt(np.mean(err**2, axis=axis))
    mse = np.mean(err**2, axis=axis)
    mae = np.mean(np.abs(err), axis=axis)
    L2_pred = np.sqrt(np.mean(u_pred**2, axis=axis))
    L2_true = np.sqrt(np.mean(u_true**2, axis=axis))
    L2_error = np.abs(L2_pred - L2_true)
    err_max = np.max(np.abs(err), axis=axis)
    err_abs = float(np.nanmax(np.abs(err)))
    max_pred = np.max(np.abs(u_pred), axis=axis)
    max_true = np.max(np.abs(u_true), axis=axis)

    return {
        "err": err,
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "err_max": err_max,
        "err_abs": err_abs,
        "L2_pred": L2_pred,
        "L2_true": L2_true,
        "L2_error": L2_error,
        "max_pred": max_pred,
        "max_true": max_true,
    }

def color_scales(u_hat, u_true):
    field_min = float(np.nanmin([np.nanmin(u_hat), np.nanmin(u_true)]))
    field_max = float(np.nanmax([np.nanmax(u_hat), np.nanmax(u_true)]))
    field_norm = colors.Normalize(vmin=field_min, vmax=field_max)
    return field_norm

def one_step_all_batches(pred,target,time):

    start_idx = 0   # 0 = first prediction in each rollout
    feature = 0     # we only have one feature in this dataset (wave height)
    uhat_1step = pred[:, start_idx, :, feature].numpy()        # (n_rollouts, node)
    utrue_1step = target[:, start_idx, :, feature].numpy()     # (n_rollouts, node)
    err_1step = uhat_1step - utrue_1step                       # (n_rollouts, node)
    abs_time = time[:, start_idx].numpy()                # (n_rollouts,)
 
    # sort by absolute time
    sort_idx = np.argsort(abs_time)
    uhat_1step = uhat_1step[sort_idx]
    utrue_1step = utrue_1step[sort_idx]
    err_1step = err_1step[sort_idx]
    abs_time = abs_time[sort_idx]

    # elapsed plotting time
    elapsed_time = abs_time - abs_time[0]

    return uhat_1step, utrue_1step, err_1step, elapsed_time
