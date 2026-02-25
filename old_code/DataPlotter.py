import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DataPlotter:
    def __init__(self, path):
        self.data = xr.open_dataset(path)
        self.tri = np.load(path.replace(".nc", "_tri.npy"))
        self.P = np.column_stack([self.data["x"], self.data["y"], self.data["z"]]).T

    def animate_sphere(self,
                       R=None,
                       out_path=None,
                       fps=10,
                       interval=100,
                       cmap_name="viridis"):

        P = np.asarray(self.P)
        tri = np.asarray(self.tri, dtype=int)
        u_data = self.data["u"].values 
        time_steps = self.data["time"].values

        # colormap and normalization computed from all data for stable colors
        u_min = float(np.nanmin(u_data))
        u_max = float(np.nanmax(u_data))
        norm = colors.Normalize(vmin=u_min, vmax=u_max)
        cmap = cm.get_cmap(cmap_name)

        tri_vertices = np.stack([P[:, tri[:, 0]].T, P[:, tri[:, 1]].T, P[:, tri[:, 2]].T], axis=1)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Build initial Poly3DCollection; facecolors will be updated
        poly = Poly3DCollection(
            tri_vertices,
            edgecolor="k",
            linewidths=0.2,
            alpha=0.95
        )
        ax.add_collection3d(poly)

        # Stable colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("u")

        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_zlim(-R, R)

        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("Distance x (m)")
        ax.set_ylabel("Distance y (m)")
        ax.set_zlabel("Distance z (m)")

        def update(frame):
            u = u_data[frame]
            # Per-face scalar by averaging its three vertex values
            u_face = u[tri].mean(axis=1)
            facecolors = cmap(norm(u_face))
            poly.set_facecolor(facecolors)
            ax.set_title(f"Wave equation on sphere (t = {time_steps[frame]:.6f})")
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
            plt.close(fig)

        return anim
    










