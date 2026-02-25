import os
import numpy as np
from shapely import points
import xarray as xr
import torch
from . import mesh_functions as mesh
from . import wave_sphere_exact as exact

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def tri_to_undirected_edges(tri):
    a = tri[:, [0, 1]]
    b = tri[:, [1, 2]]
    c = tri[:, [2, 0]]
    edges = np.vstack([a, b, c])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges  

def save_graph_from_sphere( 
    out_dir: str,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,   # (N,)
    tri: np.ndarray,                               # (T, 3)
    radius: float = 1.0,
): 
    os.makedirs(out_dir, exist_ok=True)
    N = x.shape[0]

    # Use lon/lat as 2D "positions" for compatibility with default Neural-LAM dims
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z / radius, -1.0, 1.0))
    pos2 = np.stack([lon, lat], axis=1).astype(np.float32)  # (N,2)

    # normalize positions similarly to neural-lam create_graph  
    pos_max = np.max(np.abs(pos2))
    pos2_norm = pos2 / (pos_max if pos_max > 0 else 1.0)

    torch.save([torch.tensor(pos2_norm, dtype=torch.float32)],
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

    torch.save([edge_index_m2m], os.path.join(out_dir, "m2m_edge_index.pt"))
    torch.save([m2m_feat],       os.path.join(out_dir, "m2m_features.pt"))

    # identity g2m and m2g
    idx = np.arange(N, dtype=np.int64)
    edge_index_id = torch.tensor(np.stack([idx, idx], axis=0), dtype=torch.long)
    id_feat = torch.zeros((N, 3), dtype=torch.float32)

    torch.save(edge_index_id, os.path.join(out_dir, "g2m_edge_index.pt"))
    torch.save(id_feat,       os.path.join(out_dir, "g2m_features.pt"))
    torch.save(edge_index_id, os.path.join(out_dir, "m2g_edge_index.pt"))
    torch.save(id_feat,       os.path.join(out_dir, "m2g_features.pt"))

class SimulatorWaveEquation:
    def __init__(self, R, C, Lmax, tmax, f_handle, g_handle, generations):
        self.R = R
        self.C = C
        self.Lmax = Lmax
        self.f_handle = f_handle
        self.g_handle = g_handle
        self.tmax = tmax
        self.generations = generations
        self.P, self.tri = self.create_mesh()

    def simulate(self, title, graph_name="1level"):
        u = self.data_sim_all(self.P)
        self.save_data_and_graph(u, self.tri, self.P, title, graph_name=graph_name)
 
    def save_data_and_graph(self, u, tri, P, title, graph_name="1level"):
        points = np.asarray(P.T, dtype=np.float64)  # (N,3)
        N = points.shape[0]
        time_numpy = self.create_time_steps()
        time_datetime = time_numpy #* np.timedelta64(1, "s")

        ds = xr.Dataset(
            data_vars={
                # state over time 
                "u": (("time", "grid_index"), u),

                # static features as data vars (so ml-data-prep can stack them)
                "x_static": (("grid_index",), points[:, 0].astype(np.float32)),
                "y_static": (("grid_index",), points[:, 1].astype(np.float32)),
                "z_static": (("grid_index",), points[:, 2].astype(np.float32)),
            },
            coords={
                "time": time_datetime,
                "grid_index": np.arange(N, dtype=np.int64),

                # coords for convenience/plotting (not required for ml-data-prep)
                "time": ("time", time_datetime),
                "x": ("grid_index", points[:, 0]),
                "y": ("grid_index", points[:, 1]),
                "z": ("grid_index", points[:, 2]),
            },
            attrs={
                "radius": float(self.R),
                "tmax": float(self.tmax),
            }
        )

        nc_path = f"../data/nc_files"
        os.makedirs(nc_path, exist_ok=True)
        ds.to_netcdf(os.path.join(nc_path, f"{title}.nc"))

        # Save graph next to nc in: <title>_graph/<graph_name>/
        graph_dir = os.path.join("../data/graph", f"{title}",graph_name)

        save_graph_from_sphere(
            out_dir=graph_dir, 
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            tri=np.asarray(tri, dtype=np.int64),
            radius=float(self.R),
        )

    def create_time_steps(self):
        num_frames = 100
        time_numpy = np.linspace(0.0, self.tmax, num_frames)
        return time_numpy

    def create_mesh(self):
        P, tri = mesh.get_icosahedral_mesh()
        meshes = [(P, tri)]
        for _ in range(self.generations):
            P, tri = mesh.refine_mesh(P, tri)
            meshes.append((P, tri))
        return meshes[-1]

    def data_sim(self, t, P):
        return exact.wave_sphere_exact(P.T, t, self.f_handle, self.g_handle, self.Lmax, self.C, self.R)

    def data_sim_all(self, P):
        time_steps = self.create_time_steps()
        u_list = []
        for t in time_steps:
            u_list.append(self.data_sim(t, P))
        return u_list
    
    def get_lat_long(self):
        
        points = np.asarray(self.P.T, dtype=np.float64)  # (N,3)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]    

        lon = np.arctan2(y, x)
        lat = np.arcsin(np.clip(z / self.R, -1.0, 1.0))
        pos2 = np.stack([lon, lat], axis=1).astype(np.float32)  # (N,2)

        return lat,lon,pos2,self.tri











