from neural_lam.train_model import main

args = [
"--config_path" ,"GNN_training/one_wave/yaml_files/config_wave_100_train.yaml" ,
"--graph", "GNN_training/graphs/gsub4_msub4_nn1",
"--loss", "mse",
"--seed", "42",
"--num_workers" ,"0",
"--epochs" , "100",
"--processor_layers", "1" ,
"--logger_run_name", "train_100_waves",
"--batch_size", "32" ,
"--logger-project" ,"startup_waves" ,
"--precompute_in_memory", "True",
]
            
main(args)



