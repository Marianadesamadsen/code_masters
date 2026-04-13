import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_1_step(rmse,max_err,time):

    assert len(rmse) == len(max_err) == len(time), "Input arrays must have the same length"

    fig,ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, rmse, label="RMSE")
    ax.plot(time, max_err, label="Max error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error")
    ax.legend()
    ax.set_title("1-step error over full test set")
    plt.tight_layout()
    
    return fig

def plot_error_metrics(rmse, mae, max_err, rel_rmse, time):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, rmse, label="RMSE")
    ax.plot(time, mae, label="MAE")
    ax.plot(time, max_err, label="Max error")
    ax.plot(time, rel_rmse, label="Relative RMSE")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error")
    ax.set_title("Error metrics over time")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_error_histogram(err):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(err.ravel(), bins=50)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of pointwise errors")
    fig.tight_layout()
    return fig

def plot_rmse_heatmap(pred, target):
    err = pred[..., 0].numpy() - target[..., 0].numpy()   # (n_test_data_times, T, node)
    rmse_rt = np.sqrt(np.mean(err**2, axis=2))            # (n_test_data_times, T)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rmse_rt, aspect="auto", origin="lower")
    ax.set_xlabel("rollout")
    ax.set_ylabel("Test data index")
    ax.set_title("RMSE")
    fig.colorbar(im, ax=ax, label="RMSE")
    fig.tight_layout()
    return fig

def plot_l2_energy_heatmap(pred, target):
    # Convert to numpy safely
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    # Remove feature dimension
    u_pred = pred[..., 0]    # (n_test_data_times, T, node)
    u_true = target[..., 0]

    # Compute energy per rollout and time
    e_pred = np.mean(u_pred**2, axis=2)   # (n_test_data_times, T)
    e_true = np.mean(u_true**2, axis=2)

    # Energy error
    err_energy = np.abs(e_pred - e_true)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(err_energy, aspect="auto", origin="lower")

    ax.set_xlabel("Roll out")
    ax.set_ylabel("Test data index")
    ax.set_title("Energy error heatmap")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Energy error")

    fig.tight_layout()
    return fig

def plot_rollout_error_growth(pred, target):
    err = pred[..., 0].numpy() - target[..., 0].numpy()   # (n_test_data_times, T, node)
    rmse_t = np.sqrt(np.mean(err**2, axis=(0, 2)))        # (T,)
    mae_t = np.mean(np.abs(err), axis=(0, 2))             # (T,)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rmse_t, label="RMSE")
    ax.plot(mae_t, label="MAE")
    ax.set_xlabel("Roll out index")
    ax.set_ylabel("Error")
    ax.set_title("Error growth over rollout")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_pred_vs_true_scatter(u_pred, u_true):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(u_true.ravel(), u_pred.ravel(), s=2, alpha=0.3)
    mn = min(np.min(u_true), np.min(u_pred))
    mx = max(np.max(u_true), np.max(u_pred))
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs true values")
    fig.tight_layout()
    return fig

def plot_l2_energy(pred, target):
    u_pred = pred[..., 0].numpy()    # (n_test_data_times, T, node)
    u_true = target[..., 0].numpy()

    e_pred = np.mean(u_pred**2, axis=2).mean(axis=0)
    e_true = np.mean(u_true**2, axis=2).mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(e_true, label="True")
    ax.plot(e_pred, label="Predicted")
    ax.set_xlabel("Roll out index")
    ax.set_ylabel(r"$\|u\|_2^2$")
    ax.set_title("Energy over time")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_amplitude_over_time(pred, target):
    u_pred = pred[..., 0].numpy()
    u_true = target[..., 0].numpy()

    a_pred = np.max(np.abs(u_pred), axis=2).mean(axis=0)
    a_true = np.max(np.abs(u_true), axis=2).mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(a_true, label="True")
    ax.plot(a_pred, label="Predicted")
    ax.set_xlabel("Lead time index")
    ax.set_ylabel("Max |u|")
    ax.set_title("Amplitude evolution")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_correlation_over_time(pred, target):
    u_pred = pred[..., 0].numpy()
    u_true = target[..., 0].numpy()

    corrs = []
    for t in range(u_pred.shape[1]):
        p = u_pred[:, t, :].ravel()
        q = u_true[:, t, :].ravel()
        corrs.append(np.corrcoef(p, q)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(corrs)
    ax.set_xlabel("Lead time index")
    ax.set_ylabel("Correlation")
    ax.set_title("Prediction-target correlation over time")
    fig.tight_layout()
    return fig
