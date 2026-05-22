import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import sys 
sys.path.insert(0, "./")
import scripts.PY_files.eval_models_scripts.plot_functions as plot_funcs
import xarray as xr
ds_geo_dir = "GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_15.nc"
 
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


######################### Pr wave 
n_waves = 50
n_samples = df_10_rel_error.shape[0]
samples_per_wave = n_samples//n_waves

df_10_rel_error["wave_id"] = np.arange(n_samples) // samples_per_wave
rollout_cols = [col for col in df_10_rel_error.columns if col != "wave_id"]

wave_mean_error = (df_10_rel_error.groupby("wave_id")[rollout_cols].mean())
wave_score = wave_mean_error.mean(axis=1)
worst_waves = wave_score.sort_values(ascending=False)
print(worst_waves.head(10))

plt.figure(figsize=(10, 6))
plt.imshow(
    wave_mean_error.values,
    aspect="auto",
    origin="lower",
    norm=LogNorm(vmin=1e-4, vmax=1e-1),
)
plt.colorbar(label="Relative energy error")
plt.xlabel("Rollout")
plt.ylabel("Wave")
plt.title("Mean relative energy error per initial condition")
plt.savefig("GNN_training/one_wave/different_training_size/all_results_plot/perwave_heatmaps.png")

ds = xr.open_dataset(ds_geo_dir)

centers = ds["center"].values
sigmas = ds["sigma_deg"].values
amplitudes = ds["A"].values

for wave_id in worst_waves.head(5).index:
    print(
        wave_id,
        "sigma =", sigmas[wave_id],
        "A =", amplitudes[wave_id],
        "center =", centers[wave_id],
        "score =", wave_score[wave_id],
    )

best_waves = wave_score.sort_values(ascending=True)

print("\nBest 10 waves:\n")

for wave_id in best_waves.head(5).index:
    print(
        wave_id,
        "sigma =", sigmas[wave_id],
        "A =", amplitudes[wave_id],
        "center =", centers[wave_id],
        "score =", wave_score[wave_id],
    )


# -------------------------------------------------------
# Create dataframe
# -------------------------------------------------------

df_wave = pd.DataFrame({
    "wave_id": np.arange(n_waves),
    "score": wave_score.values,
    "sigma": sigmas[50:100],
    "A": amplitudes[50:100],
    "center_x": centers[50:100, 0],
    "center_y": centers[50:100, 1],
    "center_z": centers[50:100, 2]
})

print(df_wave.head())
from scipy.stats import pearsonr

# -------------------------------------------------------
# Correlations
# -------------------------------------------------------

corr_sigma, p_sigma = pearsonr(df_wave["sigma"], df_wave["score"])
corr_A, p_A = pearsonr(df_wave["A"], df_wave["score"])

print("\nCorrelation statistics:")
print(f"score vs sigma: r = {corr_sigma:.4f}, p = {p_sigma:.4e}")
print(f"score vs A:     r = {corr_A:.4f}, p = {p_A:.4e}")

# -------------------------------------------------------
# Scatter: score vs sigma
# -------------------------------------------------------


best_idx = df_wave.nsmallest(10, "score").index
worst_idx = df_wave.nlargest(10, "score").index

fig, ax = plt.subplots(figsize=(7,5))

sc = ax.scatter(
    df_wave["sigma"],
    df_wave["score"],
    s=60,
)

# Best 5
ax.scatter(
    df_wave.loc[best_idx, "sigma"],
    df_wave.loc[best_idx, "score"],
    s=120,
    color="green",
    edgecolor="black",
    label="Best 5",
)


# Worst 5
ax.scatter(
    df_wave.loc[worst_idx, "sigma"],
    df_wave.loc[worst_idx, "score"],
    s=120,
    color="red",
    edgecolor="black",
    label="Worst 5",
)

ax.set_xlabel(r"$\sigma$ [deg]")
ax.set_ylabel("Mean relative energy error")
ax.set_title(r"Relative energy error vs $\sigma$")
ax.legend()
ax.grid(True)

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/score_vs_sigma.png",
    dpi=300,
    bbox_inches="tight",
)

# -------------------------------------------------------
# Scatter: score vs amplitude
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(7,5))

sc = ax.scatter(
    df_wave["A"],
    df_wave["score"],
    s=60,
)

# Best 5
ax.scatter(
    df_wave.loc[best_idx, "A"],
    df_wave.loc[best_idx, "score"],
    s=120,
    color="green",
    edgecolor="black",
    label="Best 5",
)

# Worst 5
ax.scatter(
    df_wave.loc[worst_idx, "A"],
    df_wave.loc[worst_idx, "score"],
    s=120,
    color="red",
    edgecolor="black",
    label="Worst 5",
)

ax.legend()
ax.set_xlabel("Amplitude A")
ax.set_ylabel("Mean relative energy error")
ax.set_title("Relative energy error vs amplitude")
ax.grid(True)

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/score_vs_amplitude.png",
    dpi=300,
    bbox_inches="tight",
)

# -------------------------------------------------------
# Histogram of scores
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(7,5))

ax.hist(
    df_wave["score"],
    bins=15,
)

ax.set_xlabel("Mean relative energy error")
ax.set_ylabel("Count")
ax.set_title("Distribution of wave energy-error scores")

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/score_histogram.png",
    dpi=300,
    bbox_inches="tight",
)

# -------------------------------------------------------
# Spatial plot of centers colored by score
# -------------------------------------------------------

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")

p = ax.scatter(
    df_wave["center_x"],
    df_wave["center_y"],
    df_wave["center_z"],
    c=df_wave["score"],
    s=80,
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title("Wave centers colored by relative energy error")

fig.colorbar(p, ax=ax, shrink=0.7, label="Mean relative energy error")

ax.set_box_aspect([1,1,1])

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/center_score_3d.png",
    dpi=300,
    bbox_inches="tight",
)



# -------------------------------------------------------
# Spatial plot of centers colored by score
# -------------------------------------------------------

best_idx = df_wave.nsmallest(5, "score").index
worst_idx = df_wave.nlargest(5, "score").index

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection="3d")

# All waves
p = ax.scatter(
    df_wave["center_x"],
    df_wave["center_y"],
    df_wave["center_z"],
    c=df_wave["score"],
    s=80,
    alpha=0.7,
)

# Best 5
ax.scatter(
    df_wave.loc[best_idx, "center_x"],
    df_wave.loc[best_idx, "center_y"],
    df_wave.loc[best_idx, "center_z"],
    color="green",
    edgecolor="black",
    s=180,
    label="Best 5",
)

# Worst 5
ax.scatter(
    df_wave.loc[worst_idx, "center_x"],
    df_wave.loc[worst_idx, "center_y"],
    df_wave.loc[worst_idx, "center_z"],
    color="red",
    edgecolor="black",
    s=180,
    label="Worst 5",
)

# Optional labels
for idx in best_idx:
    ax.text(
        df_wave.loc[idx, "center_x"],
        df_wave.loc[idx, "center_y"],
        df_wave.loc[idx, "center_z"],
        str(df_wave.loc[idx, "wave_id"]),
        fontsize=8,
    )

for idx in worst_idx:
    ax.text(
        df_wave.loc[idx, "center_x"],
        df_wave.loc[idx, "center_y"],
        df_wave.loc[idx, "center_z"],
        str(df_wave.loc[idx, "wave_id"]),
        fontsize=8,
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title("Wave centers colored by relative energy error")

fig.colorbar(
    p,
    ax=ax,
    shrink=0.7,
    label="Mean relative energy error",
)

ax.legend()

ax.set_box_aspect([1,1,1])

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/center_score_3d_bestworst.png",
    dpi=300,
    bbox_inches="tight",
)

# ============================================================
# Compare all training sizes in subplot layout
# ============================================================

training_sizes = [10, 25, 50, 100]

dfs_rmse = {
    10: df_10_rmse.copy(),
    25: df_25_rmse.copy(),
    50: df_50_rmse.copy(),
    100: df_100_rmse.copy(),
}

# -------------------------------------------------------
# Build per-wave dataframe for each training size
# -------------------------------------------------------

wave_data = {}

for train_size, df_rmse in dfs_rmse.items():

    n_waves = 50
    n_samples = df_rmse.shape[0]
    samples_per_wave = n_samples // n_waves

    df_rmse["wave_id"] = np.arange(n_samples) // samples_per_wave

    rollout_cols = [
        col for col in df_rmse.columns
        if col != "wave_id"
    ]

    wave_mean_error = (
        df_rmse
        .groupby("wave_id")[rollout_cols]
        .mean()
    )

    wave_score = wave_mean_error.mean(axis=1)

    df_wave = pd.DataFrame({
        "wave_id": np.arange(n_waves),
        "score": wave_score.values,
        "sigma": sigmas[50:100],
        "A": amplitudes[50:100],
        "center_x": centers[50:100, 0],
        "center_y": centers[50:100, 1],
        "center_z": centers[50:100, 2],
    })

    wave_data[train_size] = df_wave

# ============================================================
# Subplot: score vs sigma
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, train_size in zip(axes, training_sizes):

    df_wave = wave_data[train_size]

    best_idx = df_wave.nsmallest(10, "score").index
    worst_idx = df_wave.nlargest(10, "score").index

    # all waves
    ax.scatter(
        df_wave["sigma"],
        df_wave["score"],
        s=50,
        alpha=0.7,
    )

    # best 5
    ax.scatter(
        df_wave.loc[best_idx, "sigma"],
        df_wave.loc[best_idx, "score"],
        color="green",
        edgecolor="black",
        s=120,
        label="Best 5",
    )

    # worst 5
    ax.scatter(
        df_wave.loc[worst_idx, "sigma"],
        df_wave.loc[worst_idx, "score"],
        color="red",
        edgecolor="black",
        s=120,
        label="Worst 5",
    )

    ax.set_title(f"Training size = {train_size}")
    ax.set_xlabel(r"$\sigma$ [deg]")
    ax.set_ylabel("Mean RMSE")
    ax.grid(True)

axes[0].legend()

fig.suptitle("RMSE score vs Gaussian width", fontsize=16)

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/subplots_sigma.png",
    dpi=300,
    bbox_inches="tight",
)

# ============================================================
# Subplot: score vs amplitude
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, train_size in zip(axes, training_sizes):

    df_wave = wave_data[train_size]

    best_idx = df_wave.nsmallest(10, "score").index
    worst_idx = df_wave.nlargest(10, "score").index

    # all waves
    ax.scatter(
        df_wave["A"],
        df_wave["score"],
        s=50,
        alpha=0.7,
    )

    # best 5
    ax.scatter(
        df_wave.loc[best_idx, "A"],
        df_wave.loc[best_idx, "score"],
        color="green",
        edgecolor="black",
        s=120,
        label="Best 5",
    )

    # worst 5
    ax.scatter(
        df_wave.loc[worst_idx, "A"],
        df_wave.loc[worst_idx, "score"],
        color="red",
        edgecolor="black",
        s=120,
        label="Worst 5",
    )

    ax.set_title(f"Training size = {train_size}")
    ax.set_xlabel("Amplitude A")
    ax.set_ylabel("Mean RMSE")
    ax.grid(True)

axes[0].legend()

fig.suptitle("RMSE score vs amplitude", fontsize=16)

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/subplots_amplitude.png",
    dpi=300,
    bbox_inches="tight",
)

# ============================================================
# Subplot: center locations
# ============================================================

fig = plt.figure(figsize=(14, 12))

for i, train_size in enumerate(training_sizes):

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")

    df_wave = wave_data[train_size]

    best_idx = df_wave.nsmallest(10, "score").index
    worst_idx = df_wave.nlargest(10, "score").index

    p = ax.scatter(
        df_wave["center_x"],
        df_wave["center_y"],
        df_wave["center_z"],
        c=df_wave["score"],
        s=70,
        alpha=0.7,
    )

    # best 5
    ax.scatter(
        df_wave.loc[best_idx, "center_x"],
        df_wave.loc[best_idx, "center_y"],
        df_wave.loc[best_idx, "center_z"],
        color="green",
        edgecolor="black",
        s=180,
        label="Best 5",
    )

    # worst 5
    ax.scatter(
        df_wave.loc[worst_idx, "center_x"],
        df_wave.loc[worst_idx, "center_y"],
        df_wave.loc[worst_idx, "center_z"],
        color="red",
        edgecolor="black",
        s=180,
        label="Worst 5",
    )

    ax.set_title(f"Training size = {train_size}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect([1,1,1])

fig.colorbar(
    p,
    ax=fig.axes,
    shrink=0.7,
    label="Mean RMSE",
)

fig.suptitle("Wave centers colored by RMSE score", fontsize=16)

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/subplots_centers.png",
    dpi=300,
    bbox_inches="tight",
)



training_sizes = [10, 25, 50, 75, 100]

# df_wave must already contain:
# wave_id, score, sigma, A, center_x, center_y, center_z

best10 = df_wave.nsmallest(10, "score")
worst10 = df_wave.nlargest(10, "score")

# ============================================================
# Strip plot version: easier to read for only 10 waves
# ============================================================

fig, axes = plt.subplots(
    len(training_sizes),
    2,
    figsize=(12, 3.5 * len(training_sizes)),
    sharex="col",
)

rng = np.random.default_rng(42)

for i, train_size in enumerate(training_sizes):
    train_slice = slice(100, 100 + train_size)

    train_sigma = sigmas[train_slice]
    train_A = amplitudes[train_slice]

    groups = {
        "Training": (train_sigma, train_A),
        "Best 10": (best10["sigma"].values, best10["A"].values),
        "Worst 10": (worst10["sigma"].values, worst10["A"].values),
    }

    for j, param_name in enumerate(["sigma", "A"]):
        ax = axes[i, j]

        for x_pos, (group_name, (sigma_vals, A_vals)) in enumerate(groups.items()):
            vals = sigma_vals if param_name == "sigma" else A_vals

            jitter = rng.normal(0, 0.04, size=len(vals))

            ax.scatter(
                np.full(len(vals), x_pos) + jitter,
                vals,
                alpha=0.8,
                s=50,
                label=group_name if i == 0 else None,
            )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Training", "Best 10", "Worst 10"])
        ax.grid(True, axis="y")

        if param_name == "sigma":
            ax.set_ylabel(r"$\sigma$ [deg]")
            ax.set_title(f"Training size {train_size}: sigma")
        else:
            ax.set_ylabel("Amplitude A")
            ax.set_title(f"Training size {train_size}: amplitude")

axes[0, 0].legend(fontsize=9)

fig.suptitle(
    "Initial-condition parameters for training waves vs best/worst test waves",
    fontsize=16,
)

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/training_vs_best_worst_stripplots.png",
    dpi=300,
    bbox_inches="tight",
)

# ============================================================
# Strip plot: training vs best/worst RMSE and rel energy error
# ============================================================

training_sizes = [10, 25, 50, 75, 100]

dfs_rmse = {
    10: df_10_rmse.copy(),
    25: df_25_rmse.copy(),
    50: df_50_rmse.copy(),
    75: df_75_rmse.copy(),
    100: df_100_rmse.copy(),
}

dfs_rel_energy = {
    10: df_10_rel_error.copy(),
    25: df_25_rel_error.copy(),
    50: df_50_rel_error.copy(),
    75: df_75_rel_error.copy(),
    100: df_100_rel_error.copy(),
}

def get_wave_scores(df_metric, n_waves=50):
    df_metric = df_metric.copy()
    n_samples = df_metric.shape[0]
    samples_per_wave = n_samples // n_waves

    df_metric["wave_id"] = np.arange(n_samples) // samples_per_wave

    rollout_cols = [
        col for col in df_metric.columns
        if col != "wave_id"
    ]

    wave_mean = df_metric.groupby("wave_id")[rollout_cols].mean()
    wave_score = wave_mean.mean(axis=1)

    return wave_score


fig, axes = plt.subplots(
    len(training_sizes),
    2,
    figsize=(16, 3.5 * len(training_sizes)),
    sharex=False,
)

rng = np.random.default_rng(42)

for i, train_size in enumerate(training_sizes):

    # -----------------------------
    # Training waves
    # -----------------------------
    train_slice = slice(100, 100 + train_size)

    train_sigma = sigmas[train_slice]
    train_A = amplitudes[train_slice]

    # -----------------------------
    # RMSE best/worst test waves
    # -----------------------------
    rmse_score = get_wave_scores(dfs_rmse[train_size], n_waves=50)

    rmse_best_ids = rmse_score.nsmallest(10).index.values
    rmse_worst_ids = rmse_score.nlargest(10).index.values

    # test waves are ensemble members 50:100
    rmse_best_sigma = sigmas[50:100][rmse_best_ids]
    rmse_worst_sigma = sigmas[50:100][rmse_worst_ids]

    rmse_best_A = amplitudes[50:100][rmse_best_ids]
    rmse_worst_A = amplitudes[50:100][rmse_worst_ids]

    # -----------------------------
    # Relative energy best/worst test waves
    # -----------------------------
    rel_score = get_wave_scores(dfs_rel_energy[train_size], n_waves=50)

    rel_best_ids = rel_score.nsmallest(10).index.values
    rel_worst_ids = rel_score.nlargest(10).index.values

    rel_best_sigma = sigmas[50:100][rel_best_ids]
    rel_worst_sigma = sigmas[50:100][rel_worst_ids]

    rel_best_A = amplitudes[50:100][rel_best_ids]
    rel_worst_A = amplitudes[50:100][rel_worst_ids]

    groups = {
        "Training": (train_sigma, train_A),
        "Best RMSE": (rmse_best_sigma, rmse_best_A),
        "Worst RMSE": (rmse_worst_sigma, rmse_worst_A),
        "Best Rel. energy": (rel_best_sigma, rel_best_A),
        "Worst Rel. energy": (rel_worst_sigma, rel_worst_A),
    }

    # ========================================================
    # Sigma plot
    # ========================================================
    ax = axes[i, 0]

    for x_pos, (group_name, (sigma_vals, A_vals)) in enumerate(groups.items()):
        jitter = rng.normal(0, 0.05, size=len(sigma_vals))

        ax.scatter(
            np.full(len(sigma_vals), x_pos) + jitter,
            sigma_vals,
            s=45,
            alpha=0.8,
        )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups.keys(), rotation=25, ha="right")
    ax.set_ylabel(r"$\sigma$ [deg]")
    ax.set_title(f"Training size {train_size}: Gaussian width")
    ax.grid(True, axis="y")

    # ========================================================
    # Amplitude plot
    # ========================================================
    ax = axes[i, 1]

    for x_pos, (group_name, (sigma_vals, A_vals)) in enumerate(groups.items()):
        jitter = rng.normal(0, 0.05, size=len(A_vals))

        ax.scatter(
            np.full(len(A_vals), x_pos) + jitter,
            A_vals,
            s=45,
            alpha=0.8,
        )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups.keys(), rotation=25, ha="right")
    ax.set_ylabel("Amplitude A")
    ax.set_title(f"Training size {train_size}: amplitude")
    ax.grid(True, axis="y")


fig.suptitle(
    "Initial-condition parameters for training waves and best/worst test waves",
    fontsize=16,
)

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/"
    "training_vs_best_worst_rmse_energy_stripplots.png",
    dpi=300,
    bbox_inches="tight",
)


#################################################
metrics = {
    "RMSE": dfs_rmse,
    "Relative energy error": dfs_rel_energy,
}

# rollout indices to visualize
rollout_steps = [0, 9, 18]

for metric_name, dfs_metric in metrics.items():

    fig, axes = plt.subplots(
        1,
        len(rollout_steps),
        figsize=(5.5 * len(rollout_steps), 5),
        sharey=True,
    )

    if len(rollout_steps) == 1:
        axes = [axes]

    for ax, rollout_idx in zip(axes, rollout_steps):

        plot_data = []
        labels = []

        for train_size in training_sizes:

            df_metric = dfs_metric[train_size].copy()

            n_waves = 50
            n_samples = df_metric.shape[0]
            samples_per_wave = n_samples // n_waves

            df_metric["wave_id"] = (
                np.arange(n_samples) // samples_per_wave
            )

            # column corresponding to rollout step
            rollout_col = df_metric.columns[rollout_idx]

            # mean score per wave
            wave_score = (
                df_metric
                .groupby("wave_id")[rollout_col]
                .mean()
            )

            plot_data.append(wave_score.values)
            labels.append(str(train_size))

        # ---------------------------------------------------
        # Boxplot
        # ---------------------------------------------------

        ax.boxplot(
            plot_data,
            labels=labels,
            showmeans=True,
        )

        # ---------------------------------------------------
        # Scatter points
        # ---------------------------------------------------

        for i, values in enumerate(plot_data, start=1):

            jitter = np.random.normal(
                0,
                0.04,
                size=len(values),
            )

            ax.scatter(
                np.full(len(values), i) + jitter,
                values,
                alpha=0.45,
                s=20,
            )

        ax.set_title(f"Rollout step {rollout_idx}")
        ax.set_xlabel("Training size")
        ax.grid(True, axis="y")

        #if metric_name == "Relative energy error":
        ax.set_yscale("log")

    axes[0].set_ylabel(metric_name)

    fig.suptitle(
        f"Per-wave {metric_name} distribution vs training size",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        f"GNN_training/one_wave/different_training_size/all_results_plot/"
        f"{metric_name.replace(' ', '_').lower()}_rollout_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )


# ============================================================
# Strip plot: best/worst RMSE and rel energy for several rollouts
# Same plot, colored by rollout index
# ============================================================

def get_wave_scores_at_rollout(df_metric, rollout_idx, n_waves=50):
    df_metric = df_metric.copy()

    n_samples = df_metric.shape[0]
    samples_per_wave = n_samples // n_waves

    df_metric["wave_id"] = np.arange(n_samples) // samples_per_wave

    rollout_cols = [col for col in df_metric.columns if col != "wave_id"]
    rollout_col = rollout_cols[rollout_idx]

    wave_score = (
        df_metric
        .groupby("wave_id")[rollout_col]
        .mean()
    )

    return wave_score

dfs_rmse = {
    10: df_10_rmse.copy(),
    25: df_25_rmse.copy(),
    50: df_50_rmse.copy(),
    75: df_75_rmse.copy(),
    100: df_100_rmse.copy(),
}

dfs_rel_energy = {
    10: df_10_rel_error.copy(),
    25: df_25_rel_error.copy(),
    50: df_50_rel_error.copy(),
    75: df_75_rel_error.copy(),
    100: df_100_rel_error.copy(),
}

training_sizes = [10, 50, 100]
rollout_indices = [0, 9, 17]

rollout_colors = {
    0: "tab:blue",
    9: "tab:orange",
    17: "tab:green",
}

fig, axes = plt.subplots(
    len(training_sizes),
    2,
    figsize=(17, 3.6 * len(training_sizes)),
    sharex=False,
)

rng = np.random.default_rng(42)

for i, train_size in enumerate(training_sizes):

    train_slice = slice(100, 100 + train_size)
    train_sigma = sigmas[train_slice]
    train_A = amplitudes[train_slice]

    for j, param_name in enumerate(["sigma", "A"]):
        ax = axes[i, j]

        # Training group
        vals = train_sigma if param_name == "sigma" else train_A
        jitter = rng.normal(0, 0.05, size=len(vals))

        ax.scatter(
            np.full(len(vals), 0) + jitter,
            vals,
            s=35,
            alpha=0.45,
            color="lightgray",
            edgecolor="black",
            linewidth=0.2,
            label="Training" if i == 0 and j == 0 else None,
        )

        group_names = [
            "Training",
            "Best RMSE",
            "Worst RMSE",
            "Best Rel. energy",
            "Worst Rel. energy",
        ]

        for rollout_idx in rollout_indices:
            color = rollout_colors[rollout_idx]

            # RMSE
            rmse_score = get_wave_scores_at_rollout(
                dfs_rmse[train_size],
                rollout_idx=rollout_idx,
                n_waves=50,
            )

            rmse_best_ids = rmse_score.nsmallest(10).index.values
            rmse_worst_ids = rmse_score.nlargest(10).index.values

            # Relative energy
            rel_score = get_wave_scores_at_rollout(
                dfs_rel_energy[train_size],
                rollout_idx=rollout_idx,
                n_waves=50,
            )

            rel_best_ids = rel_score.nsmallest(10).index.values
            rel_worst_ids = rel_score.nlargest(10).index.values

            groups = {
                1: rmse_best_ids,
                2: rmse_worst_ids,
                3: rel_best_ids,
                4: rel_worst_ids,
            }

            for x_pos, ids in groups.items():
                if param_name == "sigma":
                    vals = sigmas[50:100][ids]
                else:
                    vals = amplitudes[50:100][ids]

                jitter = rng.normal(0, 0.035, size=len(vals))

                ax.scatter(
                    np.full(len(vals), x_pos) + jitter,
                    vals,
                    s=45,
                    alpha=0.75,
                    color=color,
                    edgecolor="black",
                    linewidth=0.25,
                    label=f"Rollout {rollout_idx}" if i == 0 and j == 0 and x_pos == 1 else None,
                )

        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, rotation=25, ha="right")
        ax.grid(True, axis="y")

        if param_name == "sigma":
            ax.set_ylabel(r"$\sigma$ [deg]")
            ax.set_title(f"Training size {train_size}: Gaussian width")
        else:
            ax.set_ylabel("Amplitude A")
            ax.set_title(f"Training size {train_size}: amplitude")

axes[0, 0].legend(fontsize=9)

fig.suptitle(
    "Initial-condition parameters for best/worst test waves across rollout horizons",
    fontsize=16,
)

fig.tight_layout()

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/"
    "training_vs_best_worst_multiple_rollouts_stripplots.png",
    dpi=300,
    bbox_inches="tight",
)

# ============================================================
# Median + interquartile range vs training size
# One curve per rollout horizon
# ============================================================

training_sizes = [10, 25, 50, 75, 100]
rollout_indices = [0, 9, 17]

metrics = {
    "RMSE": dfs_rmse,
    "Relative energy error": dfs_rel_energy,
}

def get_wave_scores_at_rollout(df_metric, rollout_idx, n_waves=50):
    df_metric = df_metric.copy()

    n_samples = df_metric.shape[0]
    samples_per_wave = n_samples // n_waves

    df_metric["wave_id"] = np.arange(n_samples) // samples_per_wave

    rollout_cols = [col for col in df_metric.columns if col != "wave_id"]
    rollout_col = rollout_cols[rollout_idx]

    wave_score = (
        df_metric
        .groupby("wave_id")[rollout_col]
        .mean()
    )

    return wave_score


for metric_name, dfs_metric in metrics.items():

    fig, ax = plt.subplots(figsize=(8, 5))

    for rollout_idx in rollout_indices:

        medians = []
        q25s = []
        q75s = []

        for train_size in training_sizes:

            scores = get_wave_scores_at_rollout(
                dfs_metric[train_size],
                rollout_idx=rollout_idx,
                n_waves=50,
            ).values

            medians.append(np.median(scores))
            q25s.append(np.percentile(scores, 25))
            q75s.append(np.percentile(scores, 75))

        medians = np.asarray(medians)
        q25s = np.asarray(q25s)
        q75s = np.asarray(q75s)

        ax.plot(
            training_sizes,
            medians,
            marker="o",
            linewidth=2,
            label=f"Rollout {rollout_idx}",
        )

        ax.fill_between(
            training_sizes,
            q25s,
            q75s,
            alpha=0.2,
        )

    ax.set_xlabel("Training size")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Per-wave {metric_name}: median and interquartile range")
    ax.grid(True, which="both", axis="y")
    ax.legend()

    ax.set_yscale("log")

    fig.tight_layout()

    fig.savefig(
        "GNN_training/one_wave/different_training_size/all_results_plot/"
        f"{metric_name.replace(' ', '_').lower()}_median_iqr_vs_training_size.png",
        dpi=300,
        bbox_inches="tight",
    )


def compute_per_wave_metric(df_metric, n_waves=50):
    """
    Convert dataframe of shape:
        (n_samples, n_rollouts)

    into:
        (n_waves, n_rollouts)

    by averaging over samples belonging to the same wave.
    """

    df_metric = df_metric.copy()

    n_samples = df_metric.shape[0]
    samples_per_wave = n_samples // n_waves

    df_metric["wave_id"] = (
        np.arange(n_samples) // samples_per_wave
    )

    rollout_cols = [
        col for col in df_metric.columns
        if col != "wave_id"
    ]

    df_wave = (
        df_metric
        .groupby("wave_id")[rollout_cols]
        .mean()
    )

    return df_wave.values


def plot_heatmaps_per_wave(
    metric_list,
    train_sizes,
    title="Per-wave heatmaps",
    set_label="RMSE",
):

    fig, axes = plt.subplots(
        len(metric_list),
        1,
        figsize=(10, 3 * len(metric_list)),
        sharex=True,
    )

    if len(metric_list) == 1:
        axes = [axes]

    for ax, metric, train_size in zip(
        axes,
        metric_list,
        train_sizes,
    ):

        im = ax.imshow(
            metric,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            norm=LogNorm(vmin=1e-4, vmax=1e-1),
        )

        ax.set_ylabel("Wave ID", fontsize=12)

        ax.set_title(
            f"Train size: {train_size}",
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xticks(np.arange(metric.shape[1]))

    axes[-1].set_xlabel("Rollout", fontsize=12)

    fig.subplots_adjust(left=0.12)

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


# ============================================================
# Compute per-wave RMSE
# ============================================================

rmse_10_wave = compute_per_wave_metric(df_10_rmse, n_waves=50)
rmse_50_wave = compute_per_wave_metric(df_50_rmse, n_waves=50)
rmse_100_wave = compute_per_wave_metric(df_100_rmse, n_waves=50)

# ============================================================
# Plot
# ============================================================

fig = plot_heatmaps_per_wave(
    metric_list=[
        rmse_10_wave,
        rmse_50_wave,
        rmse_100_wave,
    ],
    train_sizes=[10, 50, 100],
    title="Per-wave RMSE heatmaps for different training sizes",
    set_label="RMSE",
)

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/"
    "rmse_heatmaps_per_wave.png",
    dpi=300,
    bbox_inches="tight",
)

diff_10_50 = rmse_10_wave - rmse_50_wave
diff_50_100 = rmse_50_wave - rmse_100_wave

from matplotlib.colors import SymLogNorm
from matplotlib.colors import TwoSlopeNorm

def plot_difference_heatmaps(
    diff_list,
    labels,
    title="Difference heatmaps",
    set_label="Improvement",
):
    max_abs = max(np.nanmax(np.abs(diff)) for diff in diff_list)

    norm = SymLogNorm(
        linthresh=1e-4,
        linscale=1.0,
        vmin=-max_abs,
        vmax=max_abs,
        base=10,
    )

    fig, axes = plt.subplots(
        len(diff_list),
        1,
        figsize=(10, 3 * len(diff_list)),
        sharex=True,
    )

    if len(diff_list) == 1:
        axes = [axes]

    for ax, diff, label in zip(axes, diff_list, labels):
        im = ax.imshow(
            diff,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="coolwarm",
            norm=norm,
        )

        ax.set_ylabel("Wave ID")
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(diff.shape[1]))

    axes[-1].set_xlabel("Rollout")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(set_label)

    fig.suptitle(title, fontsize=18)

    fig.tight_layout()

    return fig


diff_10_50 = rmse_10_wave - rmse_50_wave
diff_50_100 = rmse_50_wave - rmse_100_wave

fig = plot_difference_heatmaps(
    diff_list=[diff_10_50, diff_50_100],
    labels=[
        "Train 10 - Train 50",
        "Train 50 - Train 100",
    ],
    title="Per-wave RMSE improvement with more training data",
    set_label="RMSE reduction",
)

fig.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/"
    "rmse_difference_heatmaps_per_wave.png",
    dpi=300,
    bbox_inches="tight",
)



rmse_10_wave = compute_per_wave_metric(df_10_rmse, n_waves=50)
rmse_25_wave = compute_per_wave_metric(df_25_rmse, n_waves=50)
rmse_50_wave = compute_per_wave_metric(df_50_rmse, n_waves=50)
rmse_75_wave = compute_per_wave_metric(df_75_rmse, n_waves=50)
rmse_100_wave = compute_per_wave_metric(df_100_rmse, n_waves=50)

diff_10_25 = rmse_10_wave - rmse_25_wave
diff_25_50 = rmse_25_wave - rmse_50_wave
diff_50_75 = rmse_50_wave - rmse_75_wave
diff_75_100 = rmse_75_wave - rmse_100_wave
mean_10_25 = diff_10_25.mean(axis=0)
mean_25_50 = diff_25_50.mean(axis=0)
mean_50_75 = diff_50_75.mean(axis=0)
mean_75_100 = diff_75_100.mean(axis=0)

plt.figure(figsize=(7,5))

plt.plot(mean_10_25, label="10 → 25")
plt.plot(mean_25_50, label = "25 → 50")
plt.plot(mean_50_75, label = "50 → 75")
plt.plot(mean_75_100, label = "75 → 100")

plt.xlabel("Rollout")
plt.ylabel("Mean RMSE improvement")
plt.grid(True)
plt.legend()
plt.savefig("GNN_training/one_wave/different_training_size/all_results_plot/"
    "mean_RMSE_improvement.png",
    dpi=300,
    bbox_inches="tight",
)

