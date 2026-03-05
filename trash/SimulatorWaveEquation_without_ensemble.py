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

    def setup_xarray(self,u):
        
        N = self.xyz.shape[0]
        if u.ndim != 2 or u.shape[1] != N:
            raise ValueError(f"u should have shape (time, {N}); got {u.shape}")

        if self.tri.ndim != 2 or self.tri.shape[1] != 3:
            raise ValueError(f"tri should have shape (T, 3); got {self.tri.shape}")
        T = self.tri.shape[0]

        ds = xr.Dataset(
            data_vars={
                # state over time
                "u": (("time", "grid_index"), u),

                # static features as data vars
                "x_static": (("grid_index",), self.xyz[:, 0].astype(np.float32)),
                "y_static": (("grid_index",), self.xyz[:, 1].astype(np.float32)),
                "z_static": (("grid_index",), self.xyz[:, 2].astype(np.float32)),
            },
            coords={
                "time": ("time", self.time),
                "grid_index": np.arange(N, dtype=np.int64),

                # convenience coords for plotting
                "x": ("grid_index", self.xyz[:, 0]),
                "y": ("grid_index", self.xyz[:, 1]),
                "z": ("grid_index", self.xyz[:, 2]),
            },
            attrs={
                "R": float(self.R), 
                "tmax": float(self.tmax),
                "C": float(self.C),
                "Lmax": int(self.Lmax),
                "dt": float(self.dt),
                "dx": float(self.R * np.sqrt(4 * np.pi / N)),
                "P": self.P, 
                "tri": self.tri,
            }
        ) 
        return ds

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

        return lat, lon
    
