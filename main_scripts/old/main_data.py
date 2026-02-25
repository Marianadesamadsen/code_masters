import SimulatorWaveEquation as simu
import numpy as np
import DataPlotter as dp
import xarray as xr

R = 1
C = 1
Lmax = 10
tmax = 10   
generations = 4

x0,y0,z0 = R,R,R

def f_handle(x, y, z):
    # dot product with center
    dot = x*x0 + y*y0 + z*z0
    
    cos_alpha = np.clip(dot / R**2, -1.0, 1.0)

    alpha = np.arccos(cos_alpha)
     
    return np.exp(-(alpha**2) / (2*0.2**2))

g_handle = lambda x, y, z: 0 * x

sim = simu.SimulatorWaveEquation(R=1.0, C=C, Lmax=Lmax, tmax=tmax, f_handle=f_handle, g_handle=g_handle, generations=generations)
sim.simulate("wave_sphere_data", graph_name="1level")

plotter = dp.DataPlotter("wave_sphere_data.nc", "graph/1level")
plotter.animate_sphere(out_path="wave.gif")

