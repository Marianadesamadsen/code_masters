from neural_lam.train_model import main

args = [
"--config_path" ,"GNN_training/one_wave/yaml_files/config_wave_28_ts_600_g4_sigmamin_15.yaml" ,
"--graph", "GNN_training/graphs/gsub4_msub4_nn1",
"--loss", "mse",
"--seed", "42",
"--eval","test",
"--num_workers" ,"0",
"--epochs" , "100",
"--processor_layers", "10" ,
"--logger_run_name", "test_AR1_MP10",
"--load", "saved_models/ar1_in_training_mp10/min_val_loss.ckpt" ,
"--batch_size", "32" ,
"--logger-project" ,"ar_in_training" ,
"--precompute_in_memory", "True",
"--save_eval_to_zarr_path", "GNN_training/one_wave/different_AR/results_AR1_MP10",
"--ar_steps_eval", "50" ,
"--ar_steps_train","1"
]

main(args)


