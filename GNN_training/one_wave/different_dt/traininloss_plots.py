import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data

df_rel_energy_1 = pd.read_csv(
    "GNN_training/one_wave/different_dt/dt_validationlossenergy.csv"
)

df_train_loss = pd.read_csv(
    "GNN_training/one_wave/different_dt/dt_trainingloss.csv"
)

df_val_loss_1 = pd.read_csv(
    "GNN_training/one_wave/different_dt/dt_validationlossmean.csv"
)

data_sizes = ["75_doubledt","75"]

columns_df_rel_energy_1 = [
    f"train_{i} - val_batch_energy_rel_error"
    for i in data_sizes
]

columns_train_loss = [
    f"train_{i} - train_loss_step"
    for i in data_sizes
]

columns_val_loss_1 = [
    f"train_{i} - val_mean_loss_step"
    for i in data_sizes
]

num_steps = len(df_train_loss)
window = 1000  # moving average window size

epochs = np.arange(num_steps)

# Compute moving averages

rel_energy_ma = (
    df_rel_energy_1[columns_df_rel_energy_1]
    .iloc[:num_steps]
    .rolling(window=window, min_periods=1)
    .mean()
)

val_loss_ma = (
    df_val_loss_1[columns_val_loss_1]
    .iloc[:num_steps]
    .rolling(window=window, min_periods=1)
    .mean()
)

train_loss_ma = (
    df_train_loss[columns_train_loss]
    .iloc[:num_steps]
    .rolling(window=window, min_periods=1)
    .mean()
)


fig, ax = plt.subplots(2, 1, figsize=(14, 10))

ax[0].loglog(epochs, rel_energy_ma)
ax[0].set_title(
    f"Relative Energy Error",
    fontsize=18,
)
ax[0].set_ylabel("Relative Energy Error", fontsize=18)
ax[0].legend(
    ["$2\Delta t$","$\Delta t$"],
    fontsize=14,
)
ax[0].grid(True, which="both")

ax[1].loglog(epochs, val_loss_ma)
ax[1].set_title(
    f"Validation Loss",
    fontsize=18,
)
ax[1].set_xlabel("Steps", fontsize=18)
ax[1].set_ylabel("Validation Loss", fontsize=18)
ax[1].legend(
    ["$2\Delta t$","$\Delta t$"],
    fontsize=14,
)
ax[1].grid(True, which="both")

plt.suptitle(
    "Loss curves for different timesteps $\Delta t$",
    fontsize=20,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    "GNN_training/one_wave/different_dt/all_results_plot/val_loss_rollout1_moving_average.png",
    dpi=300,
)

plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(epochs, train_loss_ma)

ax.set_title(
    f"Training Loss",
    fontsize=18,
)

ax.set_xlabel("Steps", fontsize=18)
ax.set_ylabel("Training Loss", fontsize=18)

ax.legend(
    ["$2\Delta t$","$\Delta t$"],
    fontsize=14,
)

ax.grid(True, which="both")

plt.suptitle(
    "Training loss curves for different time steps $\Delta t$",
    fontsize=20,
)

plt.tight_layout()

plt.savefig(
    "GNN_training/one_wave/different_dt/all_results_plot/train_loss_rollout1_moving_average.png",
    dpi=300,
)

plt.close()

