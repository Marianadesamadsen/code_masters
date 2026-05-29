import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load MP data

df_rel_energy_mp = pd.read_csv(
    "GNN_training/one_wave/different_mp/MP_validationlossenergy.csv"
)

df_train_loss_mp = pd.read_csv(
    "GNN_training/one_wave/different_mp/MP_trainingloss.csv"
)

df_val_loss_mp = pd.read_csv(
    "GNN_training/one_wave/different_mp/MP_validationlossmean.csv"
)


# Load MP1 baseline from training-size experiment, train_75

df_rel_energy_mp1 = pd.read_csv(
    "GNN_training/one_wave/different_training_size/trainsize_validationlossenergy.csv"
)

df_train_loss_mp1 = pd.read_csv(
    "GNN_training/one_wave/different_training_size/trainsize_trainingloss.csv"
)

df_val_loss_mp1 = pd.read_csv(
    "GNN_training/one_wave/different_training_size/trainsize_validationlossmean.csv"
)


# Settings

mp_labels = ["MP1", "MP2", "MP3"]
window = 1000

# Use shortest dataframe length so all curves match
num_steps = min(
    len(df_rel_energy_mp),
    len(df_train_loss_mp),
    len(df_val_loss_mp),
    len(df_rel_energy_mp1),
    len(df_train_loss_mp1),
    len(df_val_loss_mp1),
)

epochs = np.arange(num_steps)


# Select columns

# MP1 from training-size experiment
rel_energy_mp1_col = "train_75 - val_batch_energy_rel_error"
train_loss_mp1_col = "train_75 - train_loss_step"
val_loss_mp1_col = "train_75 - val_mean_loss_step"

# MP2 and MP3 from MP experiment
rel_energy_mp_cols = [
    "train_mp2 - val_batch_energy_rel_error",
    "train_mp3 - val_batch_energy_rel_error",
]

train_loss_mp_cols = [
    "train_mp2 - train_loss_step",
    "train_mp3 - train_loss_step",
]

val_loss_mp_cols = [
    "train_mp2 - val_mean_loss_step",
    "train_mp3 - val_mean_loss_step",
]


# Combine MP1, MP2, MP3 into common dataframes
rel_energy_all = pd.concat(
    [
        df_rel_energy_mp1[[rel_energy_mp1_col]].iloc[:num_steps].rename(
            columns={rel_energy_mp1_col: "MP1"}
        ),
        df_rel_energy_mp[rel_energy_mp_cols].iloc[:num_steps].rename(
            columns={
                rel_energy_mp_cols[0]: "MP2",
                rel_energy_mp_cols[1]: "MP3",
            }
        ),
    ],
    axis=1,
)

val_loss_all = pd.concat(
    [
        df_val_loss_mp1[[val_loss_mp1_col]].iloc[:num_steps].rename(
            columns={val_loss_mp1_col: "MP1"}
        ),
        df_val_loss_mp[val_loss_mp_cols].iloc[:num_steps].rename(
            columns={
                val_loss_mp_cols[0]: "MP2",
                val_loss_mp_cols[1]: "MP3",
            }
        ),
    ],
    axis=1,
)

train_loss_all = pd.concat(
    [
        df_train_loss_mp1[[train_loss_mp1_col]].iloc[:num_steps].rename(
            columns={train_loss_mp1_col: "MP1"}
        ),
        df_train_loss_mp[train_loss_mp_cols].iloc[:num_steps].rename(
            columns={
                train_loss_mp_cols[0]: "MP2",
                train_loss_mp_cols[1]: "MP3",
            }
        ),
    ],
    axis=1,
)


rel_energy_ma = rel_energy_all.rolling(window=window, min_periods=1).mean()
val_loss_ma = val_loss_all.rolling(window=window, min_periods=1).mean()
train_loss_ma = train_loss_all.rolling(window=window, min_periods=1).mean()


fig, ax = plt.subplots(2, 1, figsize=(14, 10))

ax[0].loglog(epochs, rel_energy_ma)
ax[0].set_title("Relative Energy Error", fontsize=18)
ax[0].set_ylabel("Relative Energy Error", fontsize=18)
ax[0].legend(mp_labels, fontsize=14)
ax[0].grid(True, which="both")

ax[1].loglog(epochs, val_loss_ma)
ax[1].set_title("Validation Loss", fontsize=18)
ax[1].set_xlabel("Steps", fontsize=18)
ax[1].set_ylabel("Validation Loss", fontsize=18)
ax[1].legend(mp_labels, fontsize=14)
ax[1].grid(True, which="both")

plt.suptitle("Loss curves for different message passing steps", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    "GNN_training/one_wave/different_mp/all_results_plot/mp_val_loss_moving_average.png",
    dpi=300,
)

plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(epochs, train_loss_ma)

ax.set_title("Training Loss", fontsize=18)
ax.set_xlabel("Steps", fontsize=18)
ax.set_ylabel("Training Loss", fontsize=18)
ax.legend(mp_labels, fontsize=14)
ax.grid(True, which="both")

plt.suptitle("Training loss curves for different message passing steps", fontsize=20)
plt.tight_layout()

plt.savefig(
    "GNN_training/one_wave/different_mp/all_results_plot/mp_train_loss_moving_average.png",
    dpi=300,
)

plt.close()