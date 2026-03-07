import sys
sys.path.insert(0, './')
import data_generation.SimulatorWaveEquation as simu
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import matplotlib.tri as mtri
import weather_model_graphs as wmg 
from data_generation import DataPlotter as dp

R = 1 # Radius
C = 1 # Wave speed 
Lmax = 5 # Maximum degree of spherical harmonics 
tmax = 10 # Maximum time  
generations = 0 # level of refinement for the grid
x0,y0,z0 = R,R,R # Initial position of the gaussian pulse
omega = C/R*np.sqrt(Lmax*(Lmax+1)) 
T_period = 2*np.pi/omega
dt = T_period/10 # Time step 
print("dt =", dt) 


# Initial condition: Gaussian pulse centered at (x0, y0, z0)
def f_handle(x, y, z):
    # dot product with center
    dot = x*x0 + y*y0 + z*z0
    
    cos_alpha = np.clip(dot / R**2, -1.0, 1.0)

    alpha = np.arccos(cos_alpha)
     
    return np.exp(-(alpha**2) / (2*0.2**2))

# Initital condition: zero initial velocity
g_handle = lambda x, y, z: 0 * x

sim = simu.SimulatorWaveEquation(R=1.0, C=C, Lmax=Lmax, tmax=tmax, f_handle=f_handle, g_handle=g_handle, generations=generations,dt=dt)
ds = sim.simulate(title=None, savedata=False, savegraph=False)  # (time, N) 

long = ds['lon'].values
lat = ds['lat'].values
tri = ds['tri'].values

coords = np.stack([long, lat], axis=1)   # (N,2) 

graph_keisler = wmg.create.archetype.create_keisler_graph(coords, mesh_node_distance=0.5,coords_crs=ccrs.PlateCarree(),graph_crs=ccrs.PlateCarree())
fig, ax = plt.subplots(figsize=(15, 9), subplot_kw={"projection": ccrs.PlateCarree()})
wmg.visualise.nx_draw_with_pos_and_attr(
    graph_keisler, ax=ax, node_size=30, edge_color_attr="component", node_color_attr="type"
)

graph_components = wmg.split_graph_by_edge_attribute(graph=graph_keisler, attr="component")

n_components = len(graph_components)
fig, axes = plt.subplots(nrows=n_components, ncols=1, figsize=(10, 9 * n_components), subplot_kw={"projection": ccrs.PlateCarree()})

for (name, g), ax in zip(graph_components.items(), axes.flatten()):
    pl_kwargs = {}
    if name == "m2m":
        pl_kwargs = dict(edge_color_attr="len") 
    elif name == "g2m" or name == "m2g":
        pl_kwargs = dict(edge_color_attr="len", node_color_attr="type")

    wmg.visualise.nx_draw_with_pos_and_attr(graph=g, ax=ax, **pl_kwargs)
    ax.set_title(name)
    ax.set_aspect(1.0)

plt.show()


