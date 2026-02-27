import numpy as np
import os
import torch

def tri_to_undirected_edges(tri: np.ndarray) -> np.ndarray:
    tri = np.asarray(tri, dtype=np.int64)
    a = tri[:, [0, 1]]
    b = tri[:, [1, 2]]
    c = tri[:, [2, 0]]
    edges = np.vstack([a, b, c])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi

def save_graph_from_sphere(
    out_dir: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,          # (N,)
    tri: np.ndarray,        # (T, 3)
    radius: float = 1.0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    tri = np.asarray(tri, dtype=np.int64)

    N = x.shape[0]
    if y.shape[0] != N or z.shape[0] != N:
        raise ValueError("x, y, z must have the same length")

    # Use lon/lat as 2D positions
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z / radius, -1.0, 1.0))
    pos2 = np.stack([lon, lat], axis=1).astype(np.float32)  # (N, 2)

    # normalize positions similarly to neural-lam create_graph
    pos_max = float(np.max(np.abs(pos2)))
    pos2_norm = pos2 / (pos_max if pos_max > 0 else 1.0)

    # Save consistently as tensors (NOT lists)
    torch.save(torch.tensor(pos2_norm, dtype=torch.float32),
               os.path.join(out_dir, "mesh_features.pt"))

    # m2m edges from triangulation adjacency
    undirected = tri_to_undirected_edges(tri)
    src = np.concatenate([undirected[:, 0], undirected[:, 1]])
    dst = np.concatenate([undirected[:, 1], undirected[:, 0]])
    edge_index_m2m = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # edge features: [len, dx, dy] (tangent-ish)
    lon_s, lat_s = lon[src], lat[src]
    lon_t, lat_t = lon[dst], lat[dst]
    dlon = wrap_to_pi(lon_t - lon_s)
    dlat = lat_t - lat_s
    lat_m = 0.5 * (lat_s + lat_t)

    dx = radius * dlon * np.cos(lat_m)
    dy = radius * dlat
    length = np.sqrt(dx * dx + dy * dy)

    m2m_feat = torch.tensor(np.stack([length, dx, dy], axis=1), dtype=torch.float32)

    torch.save(edge_index_m2m, os.path.join(out_dir, "m2m_edge_index.pt"))
    torch.save(m2m_feat,       os.path.join(out_dir, "m2m_features.pt"))

    # identity g2m and m2g
    idx = np.arange(N, dtype=np.int64)
    edge_index_id = torch.tensor(np.stack([idx, idx], axis=0), dtype=torch.long)
    id_feat = torch.zeros((N, 3), dtype=torch.float32)

    torch.save(edge_index_id, os.path.join(out_dir, "g2m_edge_index.pt"))
    torch.save(id_feat,       os.path.join(out_dir, "g2m_features.pt"))
    torch.save(edge_index_id, os.path.join(out_dir, "m2g_edge_index.pt"))
    torch.save(id_feat,       os.path.join(out_dir, "m2g_features.pt"))
