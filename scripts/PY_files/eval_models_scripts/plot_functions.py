import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_error_metrics(rmse, mae, max_err, time):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, rmse, label="RMSE")
    ax.plot(time, mae, label="MAE")
    ax.plot(time, max_err, label="Max error")
    ax.set_xlabel("Start time (initial condition)")
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
    im = ax.pcolormesh(rmse, shading="nearest") 
    ax.set_xlabel("rollout") 
    ax.set_ylabel("Test data index") 
    ax.set_title(f"RMSE for wave") 
    fig.colorbar(im, ax=ax, label="RMSE") 
    fig.tight_layout() 
    return fig

def plot_rollout_error_growth(rmse, mae):

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rmse, label="RMSE")
    ax.plot(mae, label="MAE")
    ax.set_xlabel("AR rollout")
    ax.set_ylabel("Error")
    ax.set_title("Error growth over rollout")
    ax.legend()
    fig.tight_layout()
     
    return fig

def plot_max_over_time(a_pred, a_true):
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(a_true, label="True")
    ax.plot(a_pred, label="Predicted")
    ax.set_xlabel("AR rollout")
    ax.set_ylabel("Max |u|")
    ax.set_title("Max evolution")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_energy_over_time(E_pred, E_true):
    E_pred = np.asarray(E_pred)
    E_true = np.asarray(E_true)
    E_err = E_pred - E_true

    rollout = np.arange(len(E_pred))

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    # True energy
    axes[0].plot(rollout, E_true)
    axes[0].set_ylabel("Energy")
    axes[0].set_title("True energy")

    # Predicted energy
    axes[1].plot(rollout, E_pred)
    axes[1].semilogy("Energy")
    axes[1].set_title("Predicted energy")

    # Error
    axes[2].plor(rollout, E_err)
    axes[2].set_ylabel("Error")
    axes[2].set_xlabel("AR rollout")
    axes[2].set_title("Energy error (pred - true)")

    fig.tight_layout()
    return fig