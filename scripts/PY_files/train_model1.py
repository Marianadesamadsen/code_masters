from multiprocessing import freeze_support
from neural_lam import train_model

if __name__ == "__main__":
    freeze_support()
    train_model.main([
    "--config_path", r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\Masters\code_masters\data\config.yaml",
    "--epochs", "10",
    "--graph","graph_coarse_data",
    "--loss","mse"
    ])


