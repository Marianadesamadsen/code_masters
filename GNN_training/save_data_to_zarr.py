from mllam_data_prep import create_dataset_zarr
from pathlib import Path

create_dataset_zarr(
    Path("GNN_training\one_wave\yaml_files\wave_75_train.yaml"),
)



