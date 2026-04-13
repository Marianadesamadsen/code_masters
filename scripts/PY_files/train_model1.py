from neural_lam import train_model

if __name__ == "__main__":
    train_model.main([
    "--config_path", "/zhome/5e/a/152106/code_masters/data/config.yaml",
    "--graph","graph_coarse_data",
    "--loss","mse",
    "--seed","42",
    "--num_workers","4",
    "--epochs","40",
    "--processor_layers","1",
    "--logger_run_name","train coarse data with same mesh",
    "--precision", "32",
    "--devices","auto"
    ])

 
