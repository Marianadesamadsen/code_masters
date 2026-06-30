import numpy as np
import torch

from neural_lam.weather_dataset import WeatherDataset
from neural_lam.config import load_config_and_datastore


config_path = "GNN_training/one_wave/yaml_files/config_full_data_grid4.yaml"
time_jumps = [20,40]
ar_steps = 1

config, datastore = load_config_and_datastore(config_path=config_path)

datasets = {}
for jump in time_jumps:
    datasets[jump] = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=ar_steps,
        precompute_in_memory=True,
        time_jump=jump,
    )

# Use first dataset to get time values
time_values = datasets[time_jumps[0]].da_state.time.values
dt = time_values[1] - time_values[0]

print("Stored fine dt:", dt)
print("Checking jumps:", time_jumps)
print()

# Check same dataset index across all jumps
dataset_indices_to_check = [0, 500, 1000]

for dataset_i in dataset_indices_to_check:
    print("=" * 80)
    print("Dataset index:", dataset_i)

    for jump, ds in datasets.items():
        init, target, forcing, target_times, meta = ds[dataset_i]

        sample_idx = int(meta["sample_idx"])
        ensemble = int(meta["ensemble_member"])

        expected_indices = sample_idx + jump * np.arange(2 + ar_steps)
        expected_target_indices = expected_indices[2:]
        expected_target_times = time_values[expected_target_indices]

        returned_target_times = target_times.detach().cpu().numpy()

        print(f"\ntime_jump = {jump}")
        print("ensemble:", ensemble)
        print("sample_idx:", sample_idx)
        print("expected all indices [inputs..., targets...]:", expected_indices)
        print("expected target indices:", expected_target_indices)
        print("expected target times:", expected_target_times)
        print("returned target times:", returned_target_times)

        assert np.allclose(
            returned_target_times,
            expected_target_times,
        ), f"Mismatch for jump={jump}, dataset_i={dataset_i}"

        # For AR=1, also print physical interpretation
        input_gap = time_values[expected_indices[1]] - time_values[expected_indices[0]]
        target_gap = time_values[expected_indices[2]] - time_values[expected_indices[1]]

        print("input gap / dt :", input_gap / dt)
        print("target gap / dt:", target_gap / dt)

        assert np.isclose(input_gap / dt, jump)
        assert np.isclose(target_gap / dt, jump)

