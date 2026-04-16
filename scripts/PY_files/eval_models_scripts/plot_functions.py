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

def plot_error_metrics(rmse, mae, max_err, time):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, rmse, label="RMSE")
    ax.plot(time, mae, label="MAE")
    ax.plot(time, max_err, label="Max error")
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

def plot_rmse_heatmap(rmse):

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rmse, aspect="auto", origin="lower")
    ax.set_xlabel("rollout")
    ax.set_ylabel("Test data index")
    ax.set_title(f"RMSE for wave")
    fig.colorbar(im, ax=ax, label="RMSE")
    fig.tight_layout()

    return fig

def plot_L2_norm_heatmap(L2_error):

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(L2_error, aspect="auto", origin="lower")

    ax.set_xlabel("Roll out")
    ax.set_ylabel("Test data index")
    ax.set_title("L2 error heatmap")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("L2 error")

    fig.tight_layout()
    return fig

def plot_rollout_error_growth(rmse, mae):

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rmse, label="RMSE")
    ax.plot(mae, label="MAE")
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

def plot_L2_norm(L2_pred, L2_true):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(L2_true, label="True")
    ax.plot(L2_pred, label="Predicted")
    ax.set_xlabel("Roll out index")
    ax.set_ylabel(r"$\|u\|_2^2$")
    ax.set_title("Energy over time")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_max_over_time(a_pred, a_true):
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(a_true, label="True")
    ax.plot(a_pred, label="Predicted")
    ax.set_xlabel("roll out  index")
    ax.set_ylabel("Max |u|")
    ax.set_title("Max evolution")
    ax.legend()
    fig.tight_layout()
    return fig
