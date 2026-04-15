from neural_lam import train_model

if __name__ == "__main__":
    train_model.main([
        "--config_path", "/zhome/5e/a/152106/code_masters/data/config.yaml",
        "--graph", "graph_coarse_data",
        "--loss", "mse",
        "--eval", "test",
        "--load", "/zhome/5e/a/152106/code_masters/saved_models/train coarse data with same mesh/min_val_loss.ckpt",
        "--ar_steps_eval","10",
        "--processor_layers","1",
        "--logger_run_name","Test with zarr",
        "--save_eval_to_zarr_path","/zhome/5e/a/152106/code_masters/test_eval_with_zarr",
    ])

