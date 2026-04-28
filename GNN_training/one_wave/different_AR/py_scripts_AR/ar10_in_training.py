from neural_lam.train_model import main

args = [
    "--config_path", "data/yaml_files/faster_training_tuning/config_in_memory.yaml",
    "--graph", "data/yaml_files/faster_training_tuning/graph/graph_same_mesh_grid_1_nearest_neighbor",
    "--loss", "mse",
    "--epochs", "100",
    "--seed", "42",
    "--num_workers", "0",
    "--processor_layers", "1",
    "--logger-project", "ar_in_training",
    "--logger_run_name", "ar10_in_training",
    "--batch_size", "16",
    "--precompute_in_memory", "True",
]

args2 = [
"--config_path" ,"data/yaml_files/faster_training_tuning/config_in_memory.yaml" ,
"--graph", "data/yaml_files/faster_training_tuning/graph/graph_same_mesh_grid_1_nearest_neighbor",
"--loss", "mse",
"--seed", "42",
"--num_workers" ,"0",
"--epochs" , "100",
"--processor_layers", "1" ,
"--logger_run_name", "test_sweep_lr_bs",
"--batch_size", "32" ,
"--logger-project" ,"para_sweeping_lr_bs_test" ,
"--precompute_in_memory", "True"
]

main(args2)



