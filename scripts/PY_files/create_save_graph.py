import sys
sys.path.insert(0, './')
import data_generation_functions.SimulatorWaveEquation as simu
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import weather_model_graphs as wmg 

R = 1 # Radius
C = 1 # Wave speed 
Lmax = 5 # Maximum degree of spherical harmonics 
tmax = 10 # Maximum time  
generations = 4 # level of refinement for the grid
x0,y0,z0 = R,R,R # Initial position of the gaussian pulse
omega = C/R*np.sqrt(Lmax*(Lmax+1)) 
T_period = 2*np.pi/omega
dt = T_period/20 # Time step 
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
ds = sim.simulate(title=None, savedata=False)  # (time, N) 

long = ds['lon'].values
lat = ds['lat'].values
tri = ds['tri'].values
xyz = sim.xyz

coords = np.stack([long, lat], axis=1)   # (N,2) 

graph_keisler = wmg.create.archetype.create_keisler_graph(coords, mesh_node_distance=0.5,coords_crs=ccrs.PlateCarree(),graph_crs=ccrs.PlateCarree())
fig, ax = plt.subplots(figsize=(15, 9), subplot_kw={"projection": ccrs.PlateCarree()})
wmg.visualise.nx_draw_with_pos_and_attr(
    graph_keisler, ax=ax, node_size=10, edge_color_attr="component", node_color_attr="type"
)

graph_components = wmg.split_graph_by_edge_attribute(graph=graph_keisler, attr="component")

n_components = len(graph_components)
fig, axes = plt.subplots(nrows=n_components, ncols=1, figsize=(20, 15 * n_components), subplot_kw={"projection": ccrs.PlateCarree()})

for (name, g), ax in zip(graph_components.items(), axes.flatten()):
    pl_kwargs = {}
    if name == "m2m":
        pl_kwargs = dict(edge_color_attr="len") 
    elif name == "g2m" or name == "m2g":
        pl_kwargs = dict(edge_color_attr="len", node_color_attr="type")

    wmg.visualise.nx_draw_with_pos_and_attr(graph=g, ax=ax,node_size=10, **pl_kwargs)
    ax.set_title(name)
    ax.set_aspect(1.0)


fig.savefig("data/graph/graph_test/graph_components4sub.png")

fig, axes = plt.subplots(len(graph_components), 1, figsize=(8, 4 * len(graph_components)))

if len(graph_components) == 1:
    axes = [axes]

for ax, (name, g) in zip(axes, graph_components.items()):
    lengths = np.array([data["len"] for _, _, data in g.edges(data=True)])
    ax.hist(lengths, bins=30)
    ax.set_title(f"{name} edge lengths")
    ax.set_xlabel("length")
    ax.set_ylabel("count")

plt.tight_layout()
fig.savefig("data/yaml_files/mp_vs_wavespeed/wavespeed1/graph/edge_length_histograms_2nn.png")

wmg.save.to_neural_lam(
    graph_components,          # {"g2m": nx.DiGraph, "m2m": nx.DiGraph, "m2g": nx.DiGraph}
    output_directory = "data/yaml_files/mp_vs_wavespeed/wavespeed1/graph/same_mesh_grid_1_nearest_neighbor",  # directory to save the graph data
)


