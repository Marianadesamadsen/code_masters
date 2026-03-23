from multiprocessing import freeze_support
import argparse
import torch.serialization

from neural_lam import train_model
from neural_lam.config import NeuralLAMConfig, DatastoreSelection

torch.serialization.add_safe_globals([
    argparse.Namespace,
    NeuralLAMConfig,
    DatastoreSelection,
])

if __name__ == "__main__":
    freeze_support()
    train_model.main([
        "--config_path", r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\Masters\code_masters\data\config.yaml",
        "--epochs", "10",
        "--graph", "graph_save_test",
        "--loss", "mse",
        "--eval", "val",
        "--load", r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\Masters\code_masters\saved_models\train-graph_lam-4x64-03_23_13-3578\min_val_loss.ckpt",
        "--save_eval_to_zarr_path", "mydataset.zarr"
    ])

