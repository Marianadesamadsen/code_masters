import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data

df_rel_energy_1 = pd.read_csv(
    "GNN_training/one_wave/different_training_size/trainsize_validationlossenergy.csv"
)

df_train_loss = pd.read_csv(
    "GNN_training/one_wave/different_training_size/trainsize_trainingloss.csv"
)

df_val_loss_1 = pd.read_csv(
    "GNN_training/one_wave/different_training_size/trainsize_validationloss1.csv"
)

data_sizes = [1, 10, 25, 50, 75]

columns_df_rel_energy_1 = [
    f"train_{i} - val_batch_energy_rel_error"
    for i in data_sizes
]

columns_train_loss = [
    f"train_{i} - train_loss_step"
    for i in data_sizes
]

columns_val_loss_1 = [
    f"train_{i} - val_loss_unroll1_step"
    for i in data_sizes
]

num_steps = 35000
window = 5000#5000  # moving average window size

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
    [f"Train {i}" for i in data_sizes],
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
    [f"Train {i}" for i in data_sizes],
    fontsize=14,
)
ax[1].grid(True, which="both")

plt.suptitle(
    "Loss curves for different training sizes",
    fontsize=20,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/val_loss_rollout1_moving_average.png",
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
    [f"Train {i}" for i in data_sizes],
    fontsize=14,
)

ax.grid(True, which="both")

plt.suptitle(
    "Training loss curves for different training sizes",
    fontsize=20,
)

plt.tight_layout()

plt.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/train_loss_rollout1_moving_average.png",
    dpi=300,
)

plt.close()

##########################3 Convergence plot: best validation loss vs training size

# best_val_loss = []
# best_train_loss = []
# best_rel_energy = []

# for size in data_sizes:
#     val_col = f"train_{size} - val_loss_unroll1"
#     train_col = f"train_{size} - train_loss_epoch"
#     energy_col = f"train_{size} - val_energy_rel_error_rollout1"

#     best_val_loss.append(df_val_loss_1[val_col].min())
#     best_train_loss.append(df_train_loss[train_col].min())

#     best_epoch = df_val_loss_1[val_col].idxmin()
#     best_rel_energy.append(df_rel_energy_1.loc[best_epoch, energy_col])


# fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# ax.semilogy(data_sizes, best_val_loss, marker="o", label="Best validation MSE")
# ax.semilogy(data_sizes, best_train_loss, marker="o", label="Best training loss")

# ax.set_title("Convergence with Training Set Size", fontsize=18)
# ax.set_xlabel("Number of training waves", fontsize=16)
# ax.set_ylabel("Loss", fontsize=16)
# ax.grid(True, which="both")
# ax.legend(fontsize=14)
# ax.set_xticks(data_sizes)

# plt.tight_layout()
# plt.savefig(
#     "GNN_training/one_wave/different_training_size/all_results_plot/training_size_convergence_loss.png",
#     dpi=300
# )


# fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# ax.semilogy(data_sizes, best_rel_energy, marker="o")

# ax.set_title("Energy Error vs Training Set Size", fontsize=18)
# ax.set_xlabel("Number of training waves", fontsize=16)
# ax.set_ylabel("Relative energy error at best val epoch", fontsize=16)
# ax.grid(True, which="both")
# ax.set_xticks(data_sizes)

# plt.tight_layout()
# plt.savefig(
#     "GNN_training/one_wave/different_training_size/all_results_plot/training_size_convergence_energy.png",
#     dpi=300
# )
# plt.show()

