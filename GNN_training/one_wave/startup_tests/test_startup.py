from neural_lam.train_model import main

args = [
"--config_path" ,"GNN_training\one_wave\yaml_files\config_wave_28_ts_600_g4_sigmamin_15.yaml" ,
"--graph", "GNN_training\graphs\gsub4_msub4_nn1",
"--loss", "mse",
"--seed", "42",
"--eval","test",
"--num_workers" ,"0",
"--epochs" , "100",
"--processor_layers", "1" ,
"--logger_run_name", "test_startup_bs_32",
"--load", "saved_models\startup_bs_32\min_val_loss.ckpt" ,
"--batch_size", "32" ,
"--logger-project" ,"startup_waves" ,
"--precompute_in_memory", "True",
"--save_eval_to_zarr_path", "GNN_training/one_wave/startup_tests/results/results_ar50",
"--ar_steps_eval", "50" 
]

main(args)


