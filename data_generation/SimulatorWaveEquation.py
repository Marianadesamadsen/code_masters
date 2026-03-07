import os
import numpy as np
import xarray as xr
from . import savegraph
from . import mesh_functions as mesh
from . import wave_sphere_exact as exact

class SimulatorWaveEquation: 
    def __init__(self, R, C, Lmax, tmax, f_handle, g_handle, generations, dt: int = 0.1):
        self.R = float(R)
        self.C = float(C)
        self.Lmax = int(Lmax)
        self.f_handle = f_handle
        self.g_handle = g_handle
        self.tmax = float(tmax) 
        self.generations = int(generations)
        self.dt = float(dt)
        self.P, tri = self.create_mesh()
        self.tri = np.asarray(tri, dtype=np.int64)
        self.time = self.create_time_steps()
        self.xyz = np.asarray(self.P.T, dtype=np.float64)  # (N, 3)
        self.N = self.xyz.shape[0]
        self.edges = self.tri_to_edges()
        self.dx = self.R * np.sqrt(4 * np.pi / self.N)
        self.clf = self.dt <= (self.dx / self.C)  # CFL condition
        self.clf_value = self.C * self.dt / self.dx
        self.lat, self.lon = self.get_lat_long()

    def tri_to_edges(self):
        edges = []

        for i, j, k in self.tri:
            edges.append((i, j))
            edges.append((j, k))
            edges.append((k, i))

            edges.append((j, i))
            edges.append((k, j))
            edges.append((i, k))

        edges = np.array(edges)
        edges = np.unique(edges, axis=0)

        edge_index = edges.T
        return edge_index

    def get_u(self):
        u = self.data_sim_all()  # (time, N)
        return np.asarray(u)

    def get_spectral_coeffs(self, t=0.0):

        # Use one point just to satisfy the API; coeffs come from quadrature anyway
        XYZ_dummy = np.array([[0,-self.R,self.R]])#self.xyz[:1, :]  # (1,3)

        _, coeffs = exact.wave_sphere_exact(
            XYZ_dummy, float(t),
            self.f_handle, self.g_handle,
            self.Lmax, self.C, self.R,
            return_coeffs=True,
        )
        return coeffs  # dict with flm/glm/ulm/mvals

    def simulate(self, title="test", graph_name="1level", savedata: bool = True, savegraph: bool = True):
        u = self.get_u()  # (time, N)
        ds = self.setup_xarray(u)
        if savedata:
            self.save_data(ds, title=title)
        if savegraph:
            self.save_graph(title, graph_name=graph_name)
        return ds
    
    def simulate_ensemble(self, fg_list, title="ensemble", savedata=True):

        u = self.data_sim_all_ensemble(fg_list)  # (member, time, N)
        ds = self.setup_xarray(u)
        if savedata:
            self.save_data(ds, title=title)
        return ds
    
    def save_data(self, ds, title="test"): 
        nc_path = "../data/nc_files"
        os.makedirs(nc_path, exist_ok=True)
        ds.to_netcdf(os.path.join(nc_path, f"{title}.nc"))

    def save_graph(self, title, graph_name="1level"):
        
        graph_dir = os.path.join("../data/graph", f"{title}", graph_name)
        savegraph.save_graph_from_sphere(
            out_dir=graph_dir,
            x=self.xyz[:, 0],
            y=self.xyz[:, 1],
            z=self.xyz[:, 2],
            tri=self.tri,
            radius=float(self.R),
        )

    def setup_xarray(self, u):
        N = self.xyz.shape[0]
        Ttri = self.tri.shape[0]
        E = self.edges.shape[1]  # edges is (2, E)

        if u.ndim == 2:
            u_dims = ("time", "grid_index")
            coords = {
                "time": ("time", self.time),
                "grid_index": np.arange(N, dtype=np.int64),
            }
        elif u.ndim == 3:
            nmem = u.shape[0]
            u_dims = ("member", "time", "grid_index")
            coords = {
                "member": np.arange(nmem, dtype=np.int64),
                "time": ("time", self.time),
                "grid_index": np.arange(N, dtype=np.int64),
            }
        else:
            raise ValueError(f"u must be 2D or 3D; got {u.shape}")

        ds = xr.Dataset(
            data_vars={
                "u": (u_dims, u),

                "x_static": (("grid_index",), self.xyz[:, 0].astype(np.float32)),
                "y_static": (("grid_index",), self.xyz[:, 1].astype(np.float32)),
                "z_static": (("grid_index",), self.xyz[:, 2].astype(np.float32)),
 
                # store mesh connectivity as variables (NOT attrs)
                "tri": (("triangle", "three"), self.tri.astype(np.int64)),          # (Ttri, 3)
                "edge_index": (("two", "edge"), self.edges.astype(np.int64)),       # (2, E)

                # optional: store P as (3,N) or (N,3); pick one and be consistent
                "P": (("grid_index", "xyz"), self.xyz.astype(np.float64)),          # (N,3)

                "lat": (("grid_index",), self.lat.astype(np.float32)),
                "lon": (("grid_index",), self.lon.astype(np.float32)),
            },
            coords={
                **coords,
                "x": ("grid_index", self.xyz[:, 0]),
                "y": ("grid_index", self.xyz[:, 1]),
                "z": ("grid_index", self.xyz[:, 2]),
                "triangle": np.arange(Ttri, dtype=np.int64),
                "edge": np.arange(E, dtype=np.int64),
                "xyz": np.array(["x", "y", "z"]),
                "two": np.array([0, 1], dtype=np.int64),
                "three": np.array([0, 1, 2], dtype=np.int64),
            },
            attrs={
                "R": float(self.R),
                "tmax": float(self.tmax),
                "C": float(self.C),
                "Lmax": int(self.Lmax),
                "dt": float(self.dt),
                "dx": float(self.dx),
                "cfl_ok": bool(self.clf),
                "cfl_value": float(self.clf_value),
            }
        )
        return ds

    def data_sim_all_ensemble(self, fg_list):
        u_members = []
        for (f_i, g_i) in fg_list:
            # simulate all times for this member
            u_list = [
                exact.wave_sphere_exact(
                    self.P.T, t, f_i, g_i, self.Lmax, self.C, self.R
                )
                for t in self.time
            ]
            u_members.append(np.stack(u_list, axis=0))  # (time, N)

        return np.stack(u_members, axis=0)  # (member, time, N)

    def create_time_steps(self):
        n_steps = int(np.floor(self.tmax / self.dt)) + 1
        return np.arange(n_steps) * self.dt
    
    def create_mesh(self):
        P, tri = mesh.get_icosahedral_mesh()
        for _ in range(self.generations):
            P, tri = mesh.refine_mesh(P, tri)
        return P, tri

    def data_sim(self, t):
        return exact.wave_sphere_exact(
            self.P.T, t, self.f_handle, self.g_handle, self.Lmax, self.C, self.R)

    def data_sim_all(self):
        u_list = [self.data_sim(t) for t in self.time]
        return np.stack(u_list, axis=0)  # (time, N)

    def get_lat_long(self):
        x = self.xyz[:, 0]
        y = self.xyz[:, 1]
        z = self.xyz[:, 2]

        lon = np.arctan2(y, x)
        lat = np.arcsin(np.clip(z / self.R, -1.0, 1.0))

        return lat, lon # radians   
    
