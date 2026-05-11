import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class DataPlotter:
    def __init__(self, ds=None, nc_path=None):
        if ds is not None:
            self.data = ds
        elif nc_path is not None:
            self.data = xr.open_dataset(nc_path)
        else:
            raise ValueError("Provide either ds or nc_path.")

        P = self.data["P"].values
        self.P = P.T if P.shape[0] == 3 and P.shape[1] != 3 else P  # (N, 3)
        self.tri = np.asarray(self.data["tri"].values, dtype=int)
        self.R = float(self.data.attrs["R"])

    @staticmethod
    def animate_three_spheres(
        ds_pred,
        ds_target,
        ds_error,
        out_path=None,
        fps=10,
        interval=100,
        pred_target_cmap="viridis",
        error_cmap="coolwarm",
        pred_target_norm=None,
        error_norm=None,
        titles=("Prediction", "Target", "Error"),
        colorbar_label="u",
        elev = 20,
        azim = 160
    ):
        """
        Create one animation with 3 side-by-side sphere plots:
        prediction, target, and error.

        Parameters
        ----------
        ds_pred, ds_target, ds_error : xr.Dataset
            Datasets containing variables:
            - u: (time, grid_index)
            - time
            - P
            - tri
            - attr R
        out_path : str or None
            Output path for animation, e.g. 'anim.gif' or 'anim.mp4'
        fps : int
            Frames per second for saved animation
        interval : int
            Delay between frames in ms
        pred_target_cmap : str
            Colormap for prediction and target
        error_cmap : str
            Colormap for error
        pred_target_norm : matplotlib.colors.Normalize or None
            Shared normalization for pred/target
        error_norm : matplotlib.colors.Normalize or None
            Normalization for error
        titles : tuple[str, str, str]
            Titles for the 3 panels
        colorbar_label : str
            Label for pred/target colorbars
        """
        # Geometry from prediction dataset
        P = ds_pred["P"].values
        P_plot = P.T if P.shape[0] == 3 and P.shape[1] != 3 else P
        tri = np.asarray(ds_pred["tri"].values, dtype=int)
        R = float(ds_pred.attrs["R"])

        # Fields
        u_pred = ds_pred["u"].values
        u_target = ds_target["u"].values
        u_error = ds_error["u"].values
        time_steps = ds_pred["time"].values

        # Basic checks
        if not (u_pred.shape == u_target.shape == u_error.shape):
            raise ValueError(
                f"Shape mismatch: pred {u_pred.shape}, "
                f"target {u_target.shape}, error {u_error.shape}"
            )

        if len(time_steps) != u_pred.shape[0]:
            raise ValueError("Time dimension does not match data shape.")

        # Shared norm for pred + target
        if pred_target_norm is None:
            u_min = float(np.nanmin([u_pred.min(), u_target.min()]))
            u_max = float(np.nanmax([u_pred.max(), u_target.max()]))
            pred_target_norm = colors.Normalize(vmin=u_min, vmax=u_max)

        # Separate norm for error
        if error_norm is None:
            err_abs = float(np.nanmax(np.abs(u_error)))
            error_norm = colors.Normalize(vmin=-err_abs, vmax=err_abs)

        cmap_pred_target = cm.get_cmap(pred_target_cmap)
        cmap_error = cm.get_cmap(error_cmap)

        tri_vertices = P_plot[tri]

        fig = plt.figure(figsize=(12, 4))
        axes = [
            fig.add_subplot(1, 3, 1, projection="3d"),
            fig.add_subplot(1, 3, 2, projection="3d"),
            fig.add_subplot(1, 3, 3, projection="3d"),
        ]
        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=0.02,
            top=0.92,
            wspace=0.05
        )

        datasets = [
            (u_pred, titles[0], cmap_pred_target, pred_target_norm),
            (u_target, titles[1], cmap_pred_target, pred_target_norm),
            (u_error, titles[2], cmap_error, error_norm),
        ]

        polys = []

        for ax, (_, title, cmap, norm) in zip(axes, datasets):
            poly = Poly3DCollection(
                tri_vertices,
                edgecolor="k",
                linewidths=0.1,
                alpha=0.95,
            )
            ax.add_collection3d(poly)

            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
            ax.set_zlim(-R, R)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z",labelpad=-5)
            ax.set_title(title)

            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.07, shrink=0.75)
            cbar.set_label("error" if title.lower() == "error" else colorbar_label)

            polys.append(poly)

        def update(frame):
            current_time = float(time_steps[frame])

            for poly, (u_data, title, cmap, norm), ax in zip(polys, datasets, axes):
                u = u_data[frame]                 # (N,)
                u_face = u[tri].mean(axis=1)     # (n_triangles,)
                poly.set_facecolor(cmap(norm(u_face)))
                ax.set_title(f"{title}\n AR rollout: {current_time:.6f}") #(f"{title}\nt = {current_time:.6f} s")

                 # Rotate sphere
                ax.view_init(elev=elev, azim=azim)

            return polys

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(time_steps),
            interval=interval,
            blit=False,
        )

        if out_path:
            if out_path.lower().endswith(".gif"):
                anim.save(out_path, writer="pillow", fps=fps)
            else:
                anim.save(out_path, writer="ffmpeg", fps=fps)

        return anim