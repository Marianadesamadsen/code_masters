import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import sys 
sys.path.insert(0, "./")
import scripts.PY_files.eval_models_scripts.plot_functions as plot_funcs

# df 10
df_10_pred_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_10_results/test_energy_pred_per_sample.csv")
df_10_target_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_10_results/test_energy_target_per_sample.csv")
df_10_rel_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_10_results/test_energy_rel_error_per_sample.csv")
df_10_absolute_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_10_results/test_energy_abs_error_per_sample.csv")
df_10_mse = pd.read_csv("GNN_training/one_wave/different_training_size/test_10_results/test_mse_per_sample.csv")
df_10_rmse = pd.read_csv("GNN_training/one_wave/different_training_size/test_10_results/test_rmse_per_sample.csv") 

# df 25
df_25_pred_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_25_results/test_energy_pred_per_sample.csv")
df_25_target_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_25_results/test_energy_target_per_sample.csv")
df_25_rel_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_25_results/test_energy_rel_error_per_sample.csv")
df_25_absolute_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_25_results/test_energy_abs_error_per_sample.csv")
df_25_mse = pd.read_csv("GNN_training/one_wave/different_training_size/test_25_results/test_mse_per_sample.csv")
df_25_rmse = pd.read_csv("GNN_training/one_wave/different_training_size/test_25_results/test_rmse_per_sample.csv")

# df 50
df_50_pred_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_50_results/test_energy_pred_per_sample.csv")
df_50_target_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_50_results/test_energy_target_per_sample.csv")
df_50_rel_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_50_results/test_energy_rel_error_per_sample.csv")
df_50_absolute_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_50_results/test_energy_abs_error_per_sample.csv")
df_50_mse = pd.read_csv("GNN_training/one_wave/different_training_size/test_50_results/test_mse_per_sample.csv")
df_50_rmse = pd.read_csv("GNN_training/one_wave/different_training_size/test_50_results/test_rmse_per_sample.csv")

# df 75 
df_75_pred_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_75_results/test_energy_pred_per_sample.csv")
df_75_target_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_75_results/test_energy_target_per_sample.csv")
df_75_rel_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_75_results/test_energy_rel_error_per_sample.csv")
df_75_absolute_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_75_results/test_energy_abs_error_per_sample.csv")
df_75_mse = pd.read_csv("GNN_training/one_wave/different_training_size/test_75_results/test_mse_per_sample.csv")
df_75_rmse = pd.read_csv("GNN_training/one_wave/different_training_size/test_75_results/test_rmse_per_sample.csv")

# df 100
df_100_pred_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_100_results/test_energy_pred_per_sample.csv")
df_100_target_energy = pd.read_csv("GNN_training/one_wave/different_training_size/test_100_results/test_energy_target_per_sample.csv")
df_100_rel_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_100_results/test_energy_rel_error_per_sample.csv")
df_100_absolute_error = pd.read_csv("GNN_training/one_wave/different_training_size/test_100_results/test_energy_abs_error_per_sample.csv")
df_100_mse = pd.read_csv("GNN_training/one_wave/different_training_size/test_100_results/test_mse_per_sample.csv")
df_100_rmse = pd.read_csv("GNN_training/one_wave/different_training_size/test_100_results/test_rmse_per_sample.csv")

################ plot heatmap rmse
def plot_heatmaps(rmse_list, train_sizes, title="RMSE heatmaps for different training sizes",set_label="RMSE"):
    fig, axes = plt.subplots(
        len(rmse_list),
        1,
        figsize=(10, 3 * len(rmse_list)),
        sharex=True,
    )
    # If only one subplot
    if len(rmse_list) == 1:
        axes = [axes]
    for ax, rmse, train_size in zip(axes, rmse_list, train_sizes):

        im = ax.imshow(
            rmse,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            norm=LogNorm(vmin=1e-4, vmax=1e-1),
        )
        ax.set_ylabel("Test sample", fontsize=12)
        ax.set_title(
            f"Train size: {train_size}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(np.arange(rmse.shape[1]))

    axes[-1].set_xlabel("Rollout", fontsize=12)

    # Make room on the left for colorbar
    fig.subplots_adjust(left=0.12)
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(set_label, fontsize=12)
    
    plt.suptitle(
        title,
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout()

    return fig

fig = plot_heatmaps(
    rmse_list=[
        df_10_rmse.values,
        df_50_rmse.values,
        df_100_rmse.values,
    ],
    train_sizes=[10, 50, 100],
    title="RMSE heatmaps for different training sizes",
    set_label="RMSE"
)

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/rmse_heatmaps_all.png",
    dpi=300,
    bbox_inches="tight",
)

#### Energy relative error
fig = plot_heatmaps(
    rmse_list=[
        df_10_rel_error.values,
        df_50_rel_error.values,
        df_100_rel_error.values,
    ],
    train_sizes=[10, 50, 100],
    title="Relative energy error heatmaps for different training sizes",
    set_label="Relative energy error"
)

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/rel_error_heatmaps_all.png",
    dpi=300,
    bbox_inches="tight",
)

################# Energy drift
fig, ax = plt.subplots(2, 1, figsize=(16, 10))

rollouts = np.arange(1, 19)

drift_10 = (df_10_pred_energy.mean() - df_10_pred_energy.mean().iloc[0]) / df_10_pred_energy.mean().iloc[0]
drift_25 = (df_25_pred_energy.mean() - df_25_pred_energy.mean().iloc[0]) / df_25_pred_energy.mean().iloc[0]
drift_50 = (df_50_pred_energy.mean() - df_50_pred_energy.mean().iloc[0]) / df_50_pred_energy.mean().iloc[0]
drift_75 = (df_75_pred_energy.mean() - df_75_pred_energy.mean().iloc[0]) / df_75_pred_energy.mean().iloc[0]
drift_100 = (df_100_pred_energy.mean() - df_100_pred_energy.mean().iloc[0]) / df_100_pred_energy.mean().iloc[0]

ax[0].plot(rollouts, drift_10, label=r"$10$", linestyle="--", marker="o")
ax[0].plot(rollouts, drift_25, label=r"$25$", linestyle="--", marker="o")
ax[0].plot(rollouts, drift_50, label=r"$50$", linestyle="--", marker="o")
ax[0].plot(rollouts, drift_75, label=r"$75$", linestyle="--", marker="o")
ax[0].plot(rollouts, drift_100, label=r"$100$", linestyle="--", marker="o")

ax[0].axhline(0, color="black", linestyle="-", linewidth=1.5, label="Zero drift")
ax[0].set_ylabel("Relative energy drift", fontsize=18)
ax[0].legend(fontsize=18)
ax[0].set_xticks(rollouts)
ax[0].grid()
 
ax[1].semilogy(rollouts, df_10_rel_error.mean(), label=r"$10$", linestyle="--", marker="o")
ax[1].semilogy(rollouts, df_25_rel_error.mean(), label=r"$25$", linestyle="--", marker="o")
ax[1].semilogy(rollouts, df_50_rel_error.mean(), label=r"$50$", linestyle="--", marker="o")
ax[1].semilogy(rollouts, df_75_rel_error.mean(), label=r"$75$", linestyle="--", marker="o")
ax[1].semilogy(rollouts, df_100_rel_error.mean(), label=r"$100$", linestyle="--", marker="o")
ax[1].set_ylabel("Relative energy error", fontsize=18)
ax[1].legend(fontsize=18)
ax[1].set_xticks(rollouts)
ax[1].grid()

plt.suptitle(
    "Mean relative energy drift and relative error over rollout time",
    fontsize=20
)
plt.tight_layout(pad=3.0)
plt.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/energy_drift_over_time.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

############################### RMSE over rollout time for each training size
rollouts = np.arange(1,21)
fig,ax = plt.subplots(1,1,figsize=(16, 10))
ax.plot(rollouts, df_10_rmse.mean(),label=r"$10$", linestyle="--",marker="o")
ax.plot(rollouts, df_25_rmse.mean(),label=r"$25$", linestyle="--",marker="o")
ax.plot(rollouts, df_50_rmse.mean(),label=r"$50$", linestyle="--",marker="o")
ax.plot(rollouts, df_75_rmse.mean(), label=r"$75$", linestyle="--",marker="o")
ax.plot(rollouts, df_100_rmse.mean(), label=r"$100$", linestyle="--",marker="o")
ax.set_ylabel("RMSE", fontsize=18)
ax.legend(fontsize=18)
ax.set_xticks(rollouts)
ax.grid()
plt.suptitle("Mean RMSE over rollout time for different training sizes", fontsize=20)
plt.tight_layout(pad=3.0)
plt.savefig("GNN_training/one_wave/different_training_size/all_results_plot/rmse_over_time.png")
plt.close()

########################### One step predictions for each training size relative error heatmap

train_sizes = [10, 25, 50, 75, 100]

rel_error_matrix = np.vstack([
    df_10_rel_error.iloc[:, 0].values,
    df_25_rel_error.iloc[:, 0].values,
    df_50_rel_error.iloc[:, 0].values,
    df_75_rel_error.iloc[:, 0].values,
    df_100_rel_error.iloc[:, 0].values,
])

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
im1 = ax.imshow(
    rel_error_matrix,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    norm=LogNorm(vmin=1e-4, vmax=1e-1)
)
ax.set_title("Relative energy error", fontsize=18)
ax.set_ylabel("Training size", fontsize=16)
ax.set_yticks(np.arange(len(train_sizes)))
ax.set_yticklabels([rf"${n}$" for n in train_sizes])
cbar1 = fig.colorbar(im1, ax=ax)
cbar1.set_label("Relative Error", fontsize=14)
for y in np.arange(0.5, len(train_sizes), 1):
    ax.axhline(y, color="black", linewidth=1)
plt.suptitle(
    "One-Step relative energy error for different training sizes",
    fontsize=20
)
plt.tight_layout(pad=3.0)
plt.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/one_step_heatmaps_rel_energy.png"
)
plt.close()

########################### One step predictions for each training size RMSE heatmap

train_sizes = [10, 25, 50, 75, 100]

rmse_error_matrix = np.vstack([
    df_10_rmse.iloc[:, 0].values,
    df_25_rmse.iloc[:, 0].values,
    df_50_rmse.iloc[:, 0].values,
    df_75_rmse.iloc[:, 0].values,
    df_100_rmse.iloc[:, 0].values,
])

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
im1 = ax.imshow(
    rmse_error_matrix,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    norm=LogNorm(vmin=1e-4, vmax=1e-1)
)
ax.set_title("RMSE", fontsize=18)
ax.set_ylabel("Training Size", fontsize=16)
ax.set_yticks(np.arange(len(train_sizes)))
ax.set_yticklabels([rf"${n}$" for n in train_sizes])
cbar1 = fig.colorbar(im1, ax=ax)
cbar1.set_label("RMSE", fontsize=14)
for y in np.arange(0.5, len(train_sizes), 1):
    ax.axhline(y, color="black", linewidth=1)
plt.suptitle(
    "One-step RMSE for different training sizes",
    fontsize=20
)
plt.tight_layout(pad=3.0)
plt.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/one_step_heatmaps_rmse.png"
)
plt.close()
