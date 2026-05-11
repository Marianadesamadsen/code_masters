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

