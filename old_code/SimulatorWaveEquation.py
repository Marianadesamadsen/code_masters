import numpy as np 
import mesh_functions as mesh
import wave_sphere_exact as exact
import xarray as xr

class SimulatorWaveEquation():    
    def __init__(self, R, C, Lmax, tmax, f_handle, g_handle, generations):
        self.R = R
        self.C = C
        self.Lmax = Lmax
        self.f_handle = f_handle
        self.g_handle = g_handle
        self.tmax = tmax
        self.generations = generations

    def simulate(self,title):
        P, tri = self.create_mesh()
        u = self.data_sim_all(P)
        self.save_data(u, tri, P, title)

    def save_data(self, u, tri, P,title):
        points = np.asarray(P.T, dtype=np.float64)  
        N = points.shape[0]

        da = xr.DataArray(
            u,  # (Time, N, edge_index)
            dims=("time", "grid_index"),
            coords={
                "time": self.create_time_steps(),
                "grid_index": np.arange(N),
                "x": ("grid_index", points[:, 0]),
                "y": ("grid_index", points[:, 1]),
                "z": ("grid_index", points[:, 2]),
            },
            name="u",
        ) 

        da.to_netcdf(f"{title}.nc")
        np.save(f"{title}_tri.npy", tri) 

    def create_time_steps(self):
        num_frames = 100
        time_steps = np.linspace(0.0, self.tmax, num_frames)
        return time_steps

    def create_mesh(self):
        P, tri = mesh.get_icosahedral_mesh()
        generations = self.generations
        meshes = [(P, tri)]

        for _ in range(generations):
            P, tri = mesh.refine_mesh(P, tri)
            meshes.append((P, tri))

        P, tri = meshes[-1]
        return P, tri

    def data_sim(self, t, P):
        u = exact.wave_sphere_exact(P.T, t, self.f_handle, self.g_handle, self.Lmax, self.C, self.R) 
        return u
    
    def data_sim_all(self,P):
        time_steps = self.create_time_steps()
        u_list = []
        for t in time_steps:
            u = self.data_sim(t, P)
            u_list.append(u)
        return u_list