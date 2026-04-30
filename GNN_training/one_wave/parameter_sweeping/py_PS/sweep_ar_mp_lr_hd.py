import wandb
from neural_lam.train_model import main

run = wandb.init() 
cfg = wandb.config
 
args = [
"--config_path" ,"GNN_training/one_wave/yaml_files/config_wave_28_ts_600_g4_sigmamin_15.yaml" ,
"--graph", "GNN_training/graphs/gsub4_msub4_nn1",
"--loss", "mse",
"--seed", "42",
"--num_workers" ,"0",
"--epochs", "100",
"--processor_layers", str(cfg.processor_layers) ,
"--logger-project" ,"sweep_ar_mp_lr_hd" ,
"--logger_run_name", run.name,
"--wandb_id", run.id,
"--lr", str(cfg.lr),
"--batch_size", "32",
"--precompute_in_memory", "True",
"--ar_steps_train", str(cfg.ar_steps_train)]

main(args)

run.finish()
