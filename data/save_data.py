from mllam_data_prep import create_dataset_zarr
from pathlib import Path

create_dataset_zarr(
    Path("data\coarse_data.yaml"),
)



