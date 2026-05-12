from neural_lam.train_model import main

args = [
"--config_path" ,"GNN_training/one_wave/yaml_files/config_wave_50_train.yaml" ,
"--graph", "GNN_training/graphs/gsub4_msub4_nn1",
"--loss", "mse",
"--seed", "42",
"--num_workers" ,"0",
"--epochs" , "300",
"--processor_layers", "1" ,
"--logger_run_name", "train_50_new",
"--batch_size", "32" ,
"--logger-project" ,"different_training_size" ,
"--precompute_in_memory",
"--checkpoint_every_n_steps", "20000",
"--val_time_stride", "10",
"--max_steps", "300000",
"--val_interval", "5",
]

          
main(args)


