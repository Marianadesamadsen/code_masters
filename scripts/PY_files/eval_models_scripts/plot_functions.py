import numpy as np
import matplotlib.pyplot as plt
import torch


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
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(E_pred, label="Predicted")
    ax.plot(E_true, label="True")
    ax.set_xlabel("AR rollout")
    ax.set_ylabel("Energy")
    ax.set_title("Energy over time")
    ax.legend()
    fig.tight_layout()
    return fig
