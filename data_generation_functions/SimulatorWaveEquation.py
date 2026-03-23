import os
import numpy as np
import xarray as xr
from . import wave_sphere_exact_split as exact
import trimesh

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
        self.tri = np.asarray(tri)  # (N,3)
        self.time = self.create_time_steps() 
        self.xyz = np.asarray(self.P.T, dtype=np.float64)  # (N, 3)
        self.N = self.xyz.shape[0]
        self.edges = self.tri_to_edges() # (N,2)
        x = self.xyz[:, 0]
        y = self.xyz[:, 1]
        z = self.xyz[:, 2]

        # Vertices used to compute dx (needed in Allans code)
        vx = x[self.tri.T]
        vy = y[self.tri.T]
        vz = z[self.tri.T]

        self.dx_elem = compute_dx(vx, vy, vz)
        self.dx_true = np.min(self.dx_elem)
        self.dx_old = self.R * np.sqrt(4 * np.pi / self.N)
        self.cfl = self.dt <= (self.dx_true / self.C)  # CFL condition
        self.cfl_value = self.C * self.dt / self.dx_true
        self.lat, self.lon = self.get_lat_long()

        #### From exact solution that is only needed to be computed once rather than everytime step
        self.quad = exact.setup_quadrature(self.Lmax, self.R)
        self.eval_data = exact.prepare_evaluation_points(self.P.T, self.Lmax, self.R)
        self.Y_basis_big = np.ascontiguousarray(np.hstack(exact.precompute_Ylm_basis(self.eval_data, self.Lmax)))

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
        # Get spectral coefficients to show the decay in order to choose Lmax

        fq, gq = exact.sample_initial_data_on_quadrature(self.quad, self.f_handle, self.g_handle)
        flm, glm, mvals = exact.compute_modal_coefficients(fq, gq, self.quad, self.Lmax)
        ulm = exact.evolve_modal_coefficients(flm, glm, float(t), self.Lmax, self.C, self.R)

        return {"flm": flm, "glm": glm, "ulm": ulm, "mvals": mvals}

    def simulate(self, title="test", savedata: bool = True):
        u = self.get_u()  # (time, N)
        ds = self.setup_xarray(u)
        if savedata:
            self.save_data(ds, title=title)
        return ds
    
    def save_data(self, ds, title="test"): 
        nc_path = "../data/nc_files"
        os.makedirs(nc_path, exist_ok=True)
        ds.to_netcdf(os.path.join(nc_path, f"{title}.nc"))

    def setup_xarray(self, u, centers=None, sigmas=None, amplitudes=None):
        N = self.xyz.shape[0]
        Ttri = self.tri.shape[0]
        E = self.edges.shape[1]  # edges is (2, E)

        data_vars = {
            "x_static": (("grid_index",), self.xyz[:, 0]),
            "y_static": (("grid_index",), self.xyz[:, 1]),
            "z_static": (("grid_index",), self.xyz[:, 2]),
            "tri": (("triangle", "three"), self.tri),
            "edge_index": (("two", "edge"), self.edges),
            "P": (("grid_index", "xyz"), self.xyz),
            "lat": (("grid_index",), self.lat),
            "lon": (("grid_index",), self.lon),
        }

        if u.ndim == 2:
            u_dims = ("time", "grid_index")
            coords = {
                "time": ("time", self.time),
                "grid_index": np.arange(N),
            }
            data_vars["u"] = (u_dims, u)

        elif u.ndim == 3:
            nmem = u.shape[0]
            u_dims = ("ensemble", "time", "grid_index")
            coords = {
                "ensemble": np.arange(nmem),
                "time": ("time", self.time),
                "grid_index": np.arange(N),
            }
            data_vars["u"] = (u_dims, u)

            if centers is not None:
                centers = np.asarray(centers)
                data_vars["center"] = (("ensemble", "xyz"), centers)

            if sigmas is not None:
                sigmas = np.asarray(sigmas)
                data_vars["sigma"] = (("ensemble",), sigmas) # radians
                sigma_deg = np.rad2deg(sigmas)
                data_vars["sigma_deg"] = (("ensemble",), sigma_deg)

            if amplitudes is not None:
                amplitudes = np.asarray(amplitudes)
                data_vars["A"] = (("ensemble",), amplitudes)

        else:
            raise ValueError(f"u must be 2D or 3D; got {u.shape}")

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                **coords,
                "x": ("grid_index", self.xyz[:, 0]),
                "y": ("grid_index", self.xyz[:, 1]),
                "z": ("grid_index", self.xyz[:, 2]),
                "triangle": np.arange(Ttri),
                "edge": np.arange(E),
                "xyz": np.array(["x", "y", "z"]),
            },
            attrs={
                "R": float(self.R),
                "tmax": float(self.tmax),
                "C": float(self.C),
                "Lmax": int(self.Lmax),
                "dt": float(self.dt),
                "dx": float(self.dx_true),
                "cfl_ok": bool(self.cfl),
                "cfl_value": float(self.cfl_value),
            }
        )
        
        return ds
    
    def simulate_ensemble(
        self,
        fg_list,
        title="ensemble",
        savedata=True,
        centers=None,
        sigmas=None,
        amplitudes=None,
    ):
        u = self.data_sim_all_ensemble(fg_list)  # (member, time, N)
        ds = self.setup_xarray(
            u,
            centers=centers,
            sigmas=sigmas,
            amplitudes=amplitudes,
        )
        if savedata:
            self.save_data(ds, title=title)
        return ds, u

    def data_sim_all_ensemble(self, fg_list):
        u_members = []

        for f_i, g_i in fg_list:
            fq_i, gq_i = exact.sample_initial_data_on_quadrature(self.quad, f_i, g_i)
            flm_i, glm_i, _ = exact.compute_modal_coefficients(fq_i, gq_i, self.quad, self.Lmax)
            is_real = np.isrealobj(fq_i) and np.isrealobj(gq_i)

            u_list = [
                exact.synthesize_solution(
                    exact.evolve_modal_coefficients(flm_i, glm_i, t, self.Lmax, self.C, self.R),
                    is_real,
                    self.Y_basis_big,
                )
                for t in self.time
            ]

            u_members.append(np.stack(u_list, axis=0))

        return np.stack(u_members, axis=0)

    def create_time_steps(self):
        n_steps = int(np.floor(self.tmax / self.dt)) + 1
        return np.arange(n_steps) * self.dt
    
    def create_mesh(self):
        mesh = trimesh.creation.icosphere(subdivisions=self.generations, radius=self.R)
        P = mesh.vertices.T
        tri = mesh.faces
        return P, tri

    def data_sim(self, t, is_real, flm, glm):

        ulm = exact.evolve_modal_coefficients(flm, glm, t, self.Lmax, self.C, self.R)
        u = exact.synthesize_solution(ulm, is_real, self.Y_basis_big)
        return u

    def data_sim_all(self):
        fq, gq = exact.sample_initial_data_on_quadrature(self.quad, self.f_handle, self.g_handle)
        is_real = np.isrealobj(fq) and np.isrealobj(gq)
        flm, glm, self.mvals = exact.compute_modal_coefficients(fq, gq, self.quad, self.Lmax)
        u_list = [self.data_sim(t,is_real,flm,glm) for t in self.time]
        return np.stack(u_list, axis=0)  

    def get_lat_long(self):
        x = self.xyz[:, 0]
        y = self.xyz[:, 1]
        z = self.xyz[:, 2]

        lon = np.arctan2(y, x)
        lat = np.arcsin(np.clip(z / self.R, -1.0, 1.0))

        return np.rad2deg(lat), np.rad2deg(lon)   
    

def compute_dx(x, y, z):
    """
    Compute dx using radius of inscribed circle
    """
    
    p0 = np.stack([x[0, :], y[0, :], z[0, :]], axis=1)
    p1 = np.stack([x[1, :], y[1, :], z[1, :]], axis=1)
    p2 = np.stack([x[2, :], y[2, :], z[2, :]], axis=1)

    len1 = np.linalg.norm(p0 - p1,2, axis=1)
    len2 = np.linalg.norm(p1 - p2,2, axis=1)
    len3 = np.linalg.norm(p2 - p0,2, axis=1)

    sper = 0.5 * (len1 + len2 + len3)

    # Euclidean area of the flat triangle in 3D
    area =  np.sqrt(sper * (sper - len1) * (sper - len2) * (sper - len3))

    dtscale = area / sper   # inradius
    return dtscale