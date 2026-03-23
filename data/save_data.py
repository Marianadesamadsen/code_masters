from mllam_data_prep import create_dataset_zarr
from pathlib import Path

create_dataset_zarr(
    Path("yaml_files\coarse_data.yaml"),
)



