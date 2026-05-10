/zhome/5e/a/152106/code_masters/.venv/bin/python: Error while finding module specification for 'GNN_training/one_wave/different_AR/test_AR1_MP3.py' (ModuleNotFoundError: No module named 'GNN_training/one_wave/different_AR/test_AR1_MP3'). Try using 'GNN_training/one_wave/different_AR/test_AR1_MP3' instead of 'GNN_training/one_wave/different_AR/test_AR1_MP3.py' as the module name.
/zhome/5e/a/152106/code_masters/.venv/bin/python: Error while finding module specification for 'GNN_training/one_wave/different_AR/test_AR10_MP1.py' (ModuleNotFoundError: No module named 'GNN_training/one_wave/different_AR/test_AR10_MP1'). Try using 'GNN_training/one_wave/different_AR/test_AR10_MP1' instead of 'GNN_training/one_wave/different_AR/test_AR10_MP1.py' as the module name.
/zhome/5e/a/152106/code_masters/.venv/bin/python: Error while finding module specification for 'GNN_training/one_wave/different_AR/test_AR1_MP10.py' (ModuleNotFoundError: No module named 'GNN_training/one_wave/different_AR/test_AR1_MP10'). Try using 'GNN_training/one_wave/different_AR/test_AR1_MP10' instead of 'GNN_training/one_wave/different_AR/test_AR1_MP10.py' as the module name.
/zhome/5e/a/152106/code_masters/.venv/bin/python: Error while finding module specification for 'GNN_training/one_wave/startup_tests/train_startup.py' (ModuleNotFoundError: No module named 'GNN_training/one_wave/startup_tests/train_startup'). Try using 'GNN_training/one_wave/startup_tests/train_startup' instead of 'GNN_training/one_wave/startup_tests/train_startup.py' as the module name.
/zhome/5e/a/152106/code_masters/.venv/bin/python: Error while finding module specification for 'GNN_training/one_wave/startup_tests/train_startup.py' (ModuleNotFoundError: No module named 'GNN_training/one_wave/startup_tests/train_startup'). Try using 'GNN_training/one_wave/startup_tests/train_startup' instead of 'GNN_training/one_wave/startup_tests/train_startup.py' as the module name.
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Seed set to 42
2026-05-08 17:23:19.896 | WARNING  | neural_lam.datastore.mdp:__init__:69 - Config file has been modified since zarr was created. The old zarr archive (in GNN_training/one_wave/yaml_files/wave_10_train.zarr) will be used.To generate new zarr-archive, move the old one first.
2026-05-08 17:23:20.591 | INFO     | neural_lam.utils:log_on_rank_zero:457 - The loaded datastore contains the following features:
2026-05-08 17:23:20.591 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  state   : u
/zhome/5e/a/152106/code_masters/neural-lam/neural_lam/datastore/mdp.py:210: UserWarning: no forcing or static data found in datastore
  warnings.warn("no forcing or static data found in datastore")
2026-05-08 17:23:20.594 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  static  : x_static y_static z_static
2026-05-08 17:23:20.594 | INFO     | neural_lam.utils:log_on_rank_zero:457 - With the following splits (over time):
2026-05-08 17:23:20.658 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  train   : 100 to 109
2026-05-08 17:23:20.669 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  val     : 0 to 49
2026-05-08 17:23:20.680 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  test    : 50 to 99
2026-05-08 17:23:21.775 | INFO     | neural_lam.utils:log_on_rank_zero:457 - Loaded graph with 5124 nodes (2562 grid, 2562 mesh)
2026-05-08 17:23:21.793 | INFO     | neural_lam.utils:log_on_rank_zero:457 - Edges in subgraphs: m2m=15360, g2m=2562, m2g=2562
2026-05-08 17:23:21.805 | INFO     | neural_lam.utils:setup_training_logger:514 - Wandb resume mode: None (id: None)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /zhome/5e/a/152106/.netrc.
wandb: Currently logged in as: s201205 (s201205-danmarks-tekniske-universitet-dtu) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run iephw2kn
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in wandb/run-20260508_172326-iephw2kn
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run test_10
wandb: ⭐️ View project at https://wandb.ai/s201205-danmarks-tekniske-universitet-dtu/different_training_size_test
wandb: 🚀 View run at https://wandb.ai/s201205-danmarks-tekniske-universitet-dtu/different_training_size_test/runs/iephw2kn
/zhome/5e/a/152106/code_masters/neural-lam/neural_lam/datastore/mdp.py:295: UserWarning: no forcing or static data found in datastore
  warnings.warn("no forcing or static data found in datastore")
Restoring states from the checkpoint path at saved_models/train_10/min_val_loss-v6.ckpt
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:566: The dirpath has changed from '/zhome/5e/a/152106/code_masters/saved_models/train_10' to '/zhome/5e/a/152106/code_masters/saved_models/test_10', therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at saved_models/train_10/min_val_loss-v6.ckpt
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Seed set to 42
2026-05-08 17:29:44.083 | WARNING  | neural_lam.datastore.mdp:__init__:69 - Config file has been modified since zarr was created. The old zarr archive (in GNN_training/one_wave/yaml_files/wave_25_train.zarr) will be used.To generate new zarr-archive, move the old one first.
2026-05-08 17:29:44.267 | INFO     | neural_lam.utils:log_on_rank_zero:457 - The loaded datastore contains the following features:
2026-05-08 17:29:44.268 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  state   : u
/zhome/5e/a/152106/code_masters/neural-lam/neural_lam/datastore/mdp.py:210: UserWarning: no forcing or static data found in datastore
  warnings.warn("no forcing or static data found in datastore")
2026-05-08 17:29:44.269 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  static  : x_static y_static z_static
2026-05-08 17:29:44.269 | INFO     | neural_lam.utils:log_on_rank_zero:457 - With the following splits (over time):
2026-05-08 17:29:44.299 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  train   : 100 to 124
2026-05-08 17:29:44.311 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  val     : 0 to 49
2026-05-08 17:29:44.322 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  test    : 50 to 99
2026-05-08 17:29:44.614 | INFO     | neural_lam.utils:log_on_rank_zero:457 - Loaded graph with 5124 nodes (2562 grid, 2562 mesh)
2026-05-08 17:29:44.627 | INFO     | neural_lam.utils:log_on_rank_zero:457 - Edges in subgraphs: m2m=15360, g2m=2562, m2g=2562
2026-05-08 17:29:44.636 | INFO     | neural_lam.utils:setup_training_logger:514 - Wandb resume mode: None (id: None)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /zhome/5e/a/152106/.netrc.
wandb: Currently logged in as: s201205 (s201205-danmarks-tekniske-universitet-dtu) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run gv6qgwux
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in wandb/run-20260508_172947-gv6qgwux
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run test_25
wandb: ⭐️ View project at https://wandb.ai/s201205-danmarks-tekniske-universitet-dtu/different_training_size_test
wandb: 🚀 View run at https://wandb.ai/s201205-danmarks-tekniske-universitet-dtu/different_training_size_test/runs/gv6qgwux
/zhome/5e/a/152106/code_masters/neural-lam/neural_lam/datastore/mdp.py:295: UserWarning: no forcing or static data found in datastore
  warnings.warn("no forcing or static data found in datastore")
Restoring states from the checkpoint path at saved_models/train_25/min_val_loss-v3.ckpt
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:566: The dirpath has changed from '/zhome/5e/a/152106/code_masters/saved_models/train_25' to '/zhome/5e/a/152106/code_masters/saved_models/test_25', therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at saved_models/train_25/min_val_loss-v3.ckpt
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Seed set to 42
2026-05-09 13:37:21.114 | WARNING  | neural_lam.datastore.mdp:__init__:69 - Config file has been modified since zarr was created. The old zarr archive (in GNN_training/one_wave/yaml_files/wave_10_train.zarr) will be used.To generate new zarr-archive, move the old one first.
2026-05-09 13:37:21.291 | INFO     | neural_lam.utils:log_on_rank_zero:457 - The loaded datastore contains the following features:
2026-05-09 13:37:21.291 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  state   : u
/zhome/5e/a/152106/code_masters/neural-lam/neural_lam/datastore/mdp.py:210: UserWarning: no forcing or static data found in datastore
  warnings.warn("no forcing or static data found in datastore")
2026-05-09 13:37:21.293 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  static  : x_static y_static z_static
2026-05-09 13:37:21.293 | INFO     | neural_lam.utils:log_on_rank_zero:457 - With the following splits (over time):
2026-05-09 13:37:21.349 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  train   : 100 to 109
2026-05-09 13:37:21.358 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  val     : 0 to 49
2026-05-09 13:37:21.368 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  test    : 50 to 99
2026-05-09 13:37:22.158 | INFO     | neural_lam.utils:log_on_rank_zero:457 - Loaded graph with 5124 nodes (2562 grid, 2562 mesh)
2026-05-09 13:37:22.168 | INFO     | neural_lam.utils:log_on_rank_zero:457 - Edges in subgraphs: m2m=15360, g2m=2562, m2g=2562
2026-05-09 13:37:22.177 | INFO     | neural_lam.utils:setup_training_logger:514 - Wandb resume mode: None (id: None)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /zhome/5e/a/152106/.netrc.
wandb: Currently logged in as: s201205 (s201205-danmarks-tekniske-universitet-dtu) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run x6g5yxgk
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in wandb/run-20260509_133725-x6g5yxgk
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run test_10
wandb: ⭐️ View project at https://wandb.ai/s201205-danmarks-tekniske-universitet-dtu/different_training_size_test
wandb: 🚀 View run at https://wandb.ai/s201205-danmarks-tekniske-universitet-dtu/different_training_size_test/runs/x6g5yxgk
/zhome/5e/a/152106/code_masters/neural-lam/neural_lam/datastore/mdp.py:295: UserWarning: no forcing or static data found in datastore
  warnings.warn("no forcing or static data found in datastore")
Restoring states from the checkpoint path at saved_models/train_10/min_val_loss-v6.ckpt
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:566: The dirpath has changed from '/zhome/5e/a/152106/code_masters/saved_models/train_10' to '/zhome/5e/a/152106/code_masters/saved_models/test_10', therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at saved_models/train_10/min_val_loss-v6.ckpt
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.
/etc/profile: line 73: 0: No space left on device
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Seed set to 42
2026-05-09 13:49:33.223 | WARNING  | neural_lam.datastore.mdp:__init__:69 - Config file has been modified since zarr was created. The old zarr archive (in GNN_training/one_wave/yaml_files/wave_25_train.zarr) will be used.To generate new zarr-archive, move the old one first.
2026-05-09 13:49:33.372 | INFO     | neural_lam.utils:log_on_rank_zero:457 - The loaded datastore contains the following features:
2026-05-09 13:49:33.372 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  state   : u
/zhome/5e/a/152106/code_masters/neural-lam/neural_lam/datastore/mdp.py:210: UserWarning: no forcing or static data found in datastore
  warnings.warn("no forcing or static data found in datastore")
2026-05-09 13:49:33.373 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  static  : x_static y_static z_static
2026-05-09 13:49:33.373 | INFO     | neural_lam.utils:log_on_rank_zero:457 - With the following splits (over time):
2026-05-09 13:49:33.396 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  train   : 100 to 124
2026-05-09 13:49:33.406 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  val     : 0 to 49
2026-05-09 13:49:33.415 | INFO     | neural_lam.utils:log_on_rank_zero:457 -  test    : 50 to 99
2026-05-09 13:49:33.584 | INFO     | neural_lam.utils:log_on_rank_zero:457 - Loaded graph with 5124 nodes (2562 grid, 2562 mesh)
2026-05-09 13:49:33.588 | INFO     | neural_lam.utils:log_on_rank_zero:457 - Edges in subgraphs: m2m=15360, g2m=2562, m2g=2562
2026-05-09 13:49:33.596 | INFO     | neural_lam.utils:setup_training_logger:514 - Wandb resume mode: None (id: None)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /zhome/5e/a/152106/.netrc.
wandb: Currently logged in as: s201205 (s201205-danmarks-tekniske-universitet-dtu) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run jnxf2r22
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in wandb/run-20260509_134936-jnxf2r22
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run test_25
wandb: ⭐️ View project at https://wandb.ai/s201205-danmarks-tekniske-universitet-dtu/different_training_size_test
wandb: 🚀 View run at https://wandb.ai/s201205-danmarks-tekniske-universitet-dtu/different_training_size_test/runs/jnxf2r22
/zhome/5e/a/152106/code_masters/neural-lam/neural_lam/datastore/mdp.py:295: UserWarning: no forcing or static data found in datastore
  warnings.warn("no forcing or static data found in datastore")
Restoring states from the checkpoint path at saved_models/train_25/min_val_loss-v3.ckpt
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:566: The dirpath has changed from '/zhome/5e/a/152106/code_masters/saved_models/train_25' to '/zhome/5e/a/152106/code_masters/saved_models/test_25', therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at saved_models/train_25/min_val_loss-v3.ckpt
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/zhome/5e/a/152106/code_masters/.venv/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.
