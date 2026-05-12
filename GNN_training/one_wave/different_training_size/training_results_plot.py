import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
df_energy_drift = pd.read_csv('GNN_training/one_wave/different_training_size/wandb_energy_drift.csv')
df_rel_energy_1 = pd.read_csv('GNN_training/one_wave/different_training_size/wandb_rel_energy_1.csv')
df_train_loss = pd.read_csv('GNN_training/one_wave/different_training_size/wandb_train_loss.csv')
df_val_loss_1 = pd.read_csv('GNN_training/one_wave/different_training_size/wandb_val_loss_1.csv')

data_sizes = [10,25,50,75,100]
columns_energy_drift = [f"train_{i} - val_mean_pred_energy_drift" for i in data_sizes]
columns_df_rel_energy_1 = [f"train_{i} - val_energy_rel_error_rollout1" for i in data_sizes]
columns_train_loss = [f"train_{i} - train_loss_epoch" for i in data_sizes]
columns_val_loss_1 = [f"train_{i} - val_loss_unroll1" for i in data_sizes]

epochs = np.arange(200)
fig, ax = plt.subplots(2, 1, figsize=(14, 10))
ax[0].semilogy(epochs, df_rel_energy_1[columns_df_rel_energy_1])
ax[0].set_title('Relative Energy Error Rollout 1',fontsize=18)
ax[0].set_ylabel('Relative Energy Error',fontsize=18)
ax[0].legend([f'Train {i}' for i in data_sizes],fontsize=18)
ax[0].grid()
ax[1].semilogy(epochs, df_val_loss_1[columns_val_loss_1])
ax[1].set_title('Validation Loss Rollout 1',fontsize=18)
ax[1].set_xlabel('Epochs',fontsize=18)
ax[1].set_ylabel('Validation Loss',fontsize=18)
ax[1].legend([f'Train {i}' for i in data_sizes],fontsize=18)
ax[1].grid()
plt.tight_layout()
plt.savefig('GNN_training/one_wave/different_training_size/all_results_plot/val_loss_rollout1.png')

fig,ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(epochs, df_train_loss[columns_train_loss])
ax.set_title('Training Loss Rollout 1',fontsize=18)
ax.set_ylabel('Training Loss',fontsize=18)
ax.legend([f'Train {i}' for i in data_sizes],fontsize=18)
ax.grid()
plt.tight_layout()
plt.savefig('GNN_training/one_wave/different_training_size/all_results_plot/train_loss_rollout1.png')

##########################3 Convergence plot: best validation loss vs training size

best_val_loss = []
best_train_loss = []
best_rel_energy = []

for size in data_sizes:
    val_col = f"train_{size} - val_loss_unroll1"
    train_col = f"train_{size} - train_loss_epoch"
    energy_col = f"train_{size} - val_energy_rel_error_rollout1"

    best_val_loss.append(df_val_loss_1[val_col].min())
    best_train_loss.append(df_train_loss[train_col].min())

    best_epoch = df_val_loss_1[val_col].idxmin()
    best_rel_energy.append(df_rel_energy_1.loc[best_epoch, energy_col])


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.semilogy(data_sizes, best_val_loss, marker="o", label="Best validation MSE")
ax.semilogy(data_sizes, best_train_loss, marker="o", label="Best training loss")

ax.set_title("Convergence with Training Set Size", fontsize=18)
ax.set_xlabel("Number of training waves", fontsize=16)
ax.set_ylabel("Loss", fontsize=16)
ax.grid(True, which="both")
ax.legend(fontsize=14)
ax.set_xticks(data_sizes)

plt.tight_layout()
plt.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/training_size_convergence_loss.png",
    dpi=300
)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.semilogy(data_sizes, best_rel_energy, marker="o")

ax.set_title("Energy Error vs Training Set Size", fontsize=18)
ax.set_xlabel("Number of training waves", fontsize=16)
ax.set_ylabel("Relative energy error at best val epoch", fontsize=16)
ax.grid(True, which="both")
ax.set_xticks(data_sizes)

plt.tight_layout()
plt.savefig(
    "GNN_training/one_wave/different_training_size/all_results_plot/training_size_convergence_energy.png",
    dpi=300
)
plt.show()

