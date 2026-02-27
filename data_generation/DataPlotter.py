import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

class DataPlotter:
    def __init__(self, nc_path=None, ds=None):

        if ds is not None:
            self.data = ds
        else:
            self.data = xr.open_dataset(nc_path)

        # positions from dataset
        self.P = self.data.attrs["P"].T  # (N,3)
        self.tri = self.data.attrs["tri"]
        self.R = self.data.attrs["R"]  # scalar

    def animate_sphere(self,
                       out_path=None,
                       fps=10,
                       interval=100,
                       cmap_name="viridis"):

        tri = np.asarray(self.tri, dtype=int)
        u_data = self.data["u"].values
        time_steps = self.data["time"].values

        # stable colormap range
        u_min = float(np.nanmin(u_data))
        u_max = float(np.nanmax(u_data))
        norm = colors.Normalize(vmin=u_min, vmax=u_max)
        cmap = cm.get_cmap(cmap_name)

        # Build triangle vertex coordinates (T,3,3)
        tri_vertices = self.P[tri]  # each row is (v0,v1,v2) each v is (x,y,z)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        poly = Poly3DCollection(
            tri_vertices,
            edgecolor="k",
            linewidths=0.2,
            alpha=0.95
        )
        ax.add_collection3d(poly)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("u")

        ax.set_xlim(-self.R, self.R)
        ax.set_ylim(-self.R, self.R)
        ax.set_zlim(-self.R, self.R)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("Distance x (m)")
        ax.set_ylabel("Distance y (m)")
        ax.set_zlabel("Distance z (m)")

        def update(frame):
            u = u_data[frame]              # (N,)
            u_face = u[tri].mean(axis=1)   # (T,)
            poly.set_facecolor(cmap(norm(u_face)))
            t = time_steps[frame]
            t_sec = t #/ np.timedelta64(1, "s")
            ax.set_title(f"Wave equation on sphere (t = {float(t_sec):.6f} s)")
            return [poly]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(time_steps),
            interval=interval,
            blit=False
        )

        if out_path:
            anim.save(out_path, writer="pillow", fps=fps)

        return anim