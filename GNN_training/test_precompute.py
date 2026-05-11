from neural_lam.weather_dataset import WeatherDataset
import torch
from neural_lam.datastore.mdp import MDPDatastore


config = load_config("your_config.yaml")
datastore = MDPDatastore(config)

# Same settings as your training run
split = "train"
ar_steps = 3
num_past_forcing_steps = 1
num_future_forcing_steps = 1
load_single_member = False
standardize = True

ds_zarr = WeatherDataset(
    datastore=datastore,
    split=split,
    ar_steps=ar_steps,
    num_past_forcing_steps=num_past_forcing_steps,
    num_future_forcing_steps=num_future_forcing_steps,
    load_single_member=load_single_member,
    standardize=standardize,
    precompute_in_memory=False,
)

ds_mem = WeatherDataset(
    datastore=datastore,
    split=split,
    ar_steps=ar_steps,
    num_past_forcing_steps=num_past_forcing_steps,
    num_future_forcing_steps=num_future_forcing_steps,
    load_single_member=load_single_member,
    standardize=standardize,
    precompute_in_memory=True,
)

print("Lengths:")
print("Zarr:", len(ds_zarr))
print("Mem :", len(ds_mem))

assert len(ds_zarr) == len(ds_mem)

indices = [0, 1, 2, 10, len(ds_zarr)//2, len(ds_zarr)-1]

for idx in indices:
    print(f"\nChecking sample idx = {idx}")

    a = ds_zarr[idx]
    b = ds_mem[idx]

    for name, x, y in zip(["init", "target", "forcing", "times"], a, b):
        if torch.is_floating_point(x):
            same = torch.allclose(x, y, atol=1e-6, rtol=1e-6)
            maxdiff = (x - y).abs().max().item()
            print(f"{name:8s}: same={same}, maxdiff={maxdiff:.3e}")
        else:
            same = torch.equal(x, y)
            print(f"{name:8s}: same={same}")