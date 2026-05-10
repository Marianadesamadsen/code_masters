from neural_lam.train_model import main

args = [
"--config_path" ,"GNN_training/one_wave/yaml_files/config_wave_10_train.yaml" ,
"--graph", "GNN_training/graphs/gsub4_msub4_nn1",
"--loss", "mse",
"--seed", "42",
"--num_workers" ,"0",
"--epochs" , "200",
"--processor_layers", "1" ,
"--logger_run_name", "test_10",
"--batch_size", "32" ,
"--logger-project" ,"different_training_size_test",
"--precompute_in_memory", "True",
"--eval", "test",
"--load", "saved_models/train_10/min_val_loss-v6.ckpt",
"--ar_steps_eval", "50",
"--save_eval_to_zarr_path", "GNN_training/one_wave/different_training_size/test_10_results.zarr"
]

main(args)
