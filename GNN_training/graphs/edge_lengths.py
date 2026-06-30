import torch

GRAPH_DIR = "GNN_training/graphs/gsub4_msub1_nn_g2m91_m1g4"

edge_attr = torch.load(f"{GRAPH_DIR}/m2g_features.pt")

if isinstance(edge_attr, list):
    edge_attr = edge_attr[0]

dist = edge_attr[:, 0]

dx = 0.08262746962887216

print("edge_attr shape:", edge_attr.shape)
print("max m2m distance:", dist.max().item())
print("mean m2m distance:", dist.mean().item())
print("max in grid-dx units:", dist.max().item() / dx)
print("mean in grid-dx units:", dist.mean().item() / dx)