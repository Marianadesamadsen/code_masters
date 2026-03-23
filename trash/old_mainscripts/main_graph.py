import SimulatorWaveEquation as simu
import numpy as np
import DataPlotter as dp
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.tri as mtri
import sys
sys.path.insert(0,"./weather-model-graphs")
import src.weather_model_graphs as wmg

R = 1
C = 1
Lmax = 10
tmax = 10   
generations = 2

x0,y0,z0 = R,R,R

def f_handle(x, y, z):
    # dot product with center
    dot = x*x0 + y*y0 + z*z0
    
    cos_alpha = np.clip(dot / R**2, -1.0, 1.0)

    alpha = np.arccos(cos_alpha)
     
    return np.exp(-(alpha**2) / (2*0.2**2))

g_handle = lambda x, y, z: 0 * x

sim = simu.SimulatorWaveEquation(R=1.0, C=C, Lmax=Lmax, tmax=tmax, f_handle=f_handle, g_handle=g_handle, generations=generations)
lat, long, pos, tri = sim.get_lat_long()


### Tringulation in lat-long space
triang = mtri.Triangulation(long, lat, triangles=tri)
plt.figure(figsize=(10,5))
plt.triplot(triang, linewidth=0.3)
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi/2, np.pi/2)
plt.xlabel("lon (rad)")
plt.ylabel("lat (rad)")
plt.savefig("graph_fig_tests/lat_long_triangulation.png")
plt.close()

### Visualize lat-long points
coords = np.stack([long, lat], axis=1)   # (N,2)

fig, ax = plt.subplots(figsize=(15, 9))
ax.scatter(coords[:, 0], coords[:, 1], marker=".")
#ax.set_extent((-180, 180, -90, 90))

plt.savefig("graph_fig_tests/lat_long.png")
plt.close()

###################### Create graph ######################
graph = wmg.create.archetype.create_keisler_graph(coords, mesh_node_distance=0.5)
fig, ax = plt.subplots(figsize=(15, 9), subplot_kw={"projection": ccrs.PlateCarree()})
wmg.visualise.nx_draw_with_pos_and_attr(
    graph, ax=ax, node_size=30, edge_color_attr="component", node_color_attr="type"
)

# graph = wmg.create.archetype.create_keisler_graph(coords, mesh_node_distance=0.5)
# fig, ax = plt.subplots(figsize=(15, 9), subplot_kw={"projection": ccrs.PlateCarree()})
# wmg.visualise.nx_draw_with_pos_and_attr(
#     graph, ax=ax, node_size=30, edge_color_attr="component", node_color_attr="type"
# )
# ax.coastlines()
# ax.set_extent((-180, 180, -90, 90))

plt.savefig("graph_fig_tests/keisler_graph_withcrs.png")
plt.close()

##################### Create splitted graphs ######################
# graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attr="component")

# n_components = len(graph_components)
# fig, axes = plt.subplots(nrows=n_components, ncols=1, figsize=(10, 9 * n_components))

# for (name, g), ax in zip(graph_components.items(), axes.flatten()):
#     pl_kwargs = {}
#     if name == "m2m":
#         pl_kwargs = dict(edge_color_attr="len")
#     elif name == "g2m" or name == "m2g":
#         pl_kwargs = dict(edge_color_attr="len", node_color_attr="type")

#     wmg.visualise.nx_draw_with_pos_and_attr(graph=g, ax=ax, **pl_kwargs)
#     ax.set_title(name)
#     ax.set_aspect(1.0)

# plt.savefig("graph_fig_tests/split_graphs.png")
# plt.close()


