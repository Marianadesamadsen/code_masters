import wandb
from neural_lam.train_model import main

run = wandb.init()
cfg = wandb.config
 
args = [
    "--config_path", "data/yaml_files/faster_training_tuning/config_in_memory.yaml",
    "--graph", "/zhome/5e/a/152106/code_masters/data/yaml_files/faster_training_tuning/graph/graph_same_mesh_grid_1_nearest_neighbor",
    "--loss", "mse",
    "--epochs", "100",
    "--seed", "42",
    "--num_workers", "0",
    "--processor_layers", "1",
    "--logger-project", "parameter_sweeping",
    "--logger_run_name", run.name,
    "--wandb_id", run.id,
    "--lr", str(cfg.lr),
    "--batch_size", str(cfg.batch_size),
    "--precompute_in_memory", "True",
]


main(args)
run.finish()
