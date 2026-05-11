from neural_lam.train_model import main

args = [
    "--config_path", "GNN_training\one_wave\yaml_files\config_wave_28_ts_600_g4_sigmamin_15.yaml",
    "--graph", "GNN_training\graphs\gsub4_msub4_nn1",
    "--loss", "mse",
    "--epochs", "100",
    "--seed", "42",
    "--num_workers", "0",
    "--processor_layers", "1",
    "--logger-project", "ar_in_training",
    "--logger_run_name", "ar50_in_eval",
    "--batch_size", "16",
    "--precompute_in_memory", "True",
    "--ar_steps_eval", "50",
    "--val_steps_to_log", "1", "2", "3", "5", "10", "50"
]


main(args)



