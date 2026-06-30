import torch

ckpt = torch.load(
    "saved_models/dt10_mp1/last.ckpt",
    map_location="cpu",
    weights_only=False,
)

print("global_step:", ckpt["global_step"])
print("epoch:", ckpt.get("epoch"))
