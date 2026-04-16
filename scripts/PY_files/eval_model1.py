from neural_lam import train_model
import neural_lam

# if __name__ == "__main__":
#     train_model.main([
#         "--config_path", "/zhome/5e/a/152106/code_masters/data/config.yaml",
#         "--graph", "graph_coarse_data",
#         "--loss", "mse",
#         "--eval", "test",
#         "--load", "/zhome/5e/a/152106/code_masters/saved_models/train coarse data with same mesh/min_val_loss.ckpt",
#         "--ar_steps_eval","10",
#         "--processor_layers","1",
#         "--logger_run_name","Test with zarr",
#         "--save_eval_to_zarr_path","/zhome/5e/a/152106/code_masters/test_eval_with_zarr",
#     ])

if __name__ == "__main__":
    train_model.main([
        "--config_path", "/zhome/5e/a/152106/code_masters/data/config_10_waves.yaml",
        "--graph", "graph_same_mesh_grid_1_nearest_neighbor",
        "--loss", "mse",
        "--eval", "test",
        "--load", "/zhome/5e/a/152106/code_masters/saved_models/train 10 waves/min_val_loss.ckpt",
        "--ar_steps_eval","10",
        "--processor_layers","1",
        "--logger_run_name","test 10 waves debug",
        "--save_eval_to_zarr_path", "/zhome/5e/a/152106/code_masters/evaluation_results/waves10_test",
    ])
 