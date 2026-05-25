import os
import sys
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors, cm, animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "./")
import scripts.PY_files.eval_models_scripts.helper_functions_ensemble as helper
 

def plot_results(
    ds_geo_dir,
    raw_dirs,
    plot_dir,
    anim_dir,
    generations,
    training_size_labels,
    plot_animations_rollout=True,
    rolloutidx=150,
    azim=160,
    elev=20,
):
    ds_geo = xr.open_dataset(ds_geo_dir)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    P = ds_geo["P"].values
    tri = ds_geo["tri"].values
    R = ds_geo.attrs["R"]

    rows = []
    all_pred = []
    all_target = []
    all_error = []
    max_data = []

    for raw_dir, label in zip(raw_dirs, training_size_labels):
        ds_all = xr.open_zarr(raw_dir)

        pred_all = ds_all["prediction"].values
        target_all = ds_all["target"].values
        rollout_steps = ds_all.rollout_step.values

        pred_all_1feature = pred_all[:, :, :, 0]
        target_all_1feature = target_all[:, :, :, 0]
        error_all_1feature = pred_all_1feature - target_all_1feature
        
        max_pred = np.nanmax(pred_all_1feature[rolloutidx], axis=1)
        max_target = np.nanmax(target_all_1feature[rolloutidx], axis=1)
        max_data.append((label, rollout_steps, max_pred, max_target))

        pred_rollout = pred_all_1feature[rolloutidx]
        target_rollout = target_all_1feature[rolloutidx]
        error_rollout = error_all_1feature[rolloutidx]

        ds_pred = helper.setup_simple_xarray(pred_rollout, rollout_steps, P, tri, R=R)
        ds_true = helper.setup_simple_xarray(target_rollout, rollout_steps, P, tri, R=R)
        ds_err = helper.setup_simple_xarray(error_rollout, rollout_steps, P, tri, R=R)

        rows.append((ds_pred, ds_true, ds_err, label))

        all_pred.append(pred_rollout)
        all_target.append(target_rollout)
        all_error.append(error_rollout)

    u_min = float(np.nanmin([np.nanmin(arr) for arr in all_pred + all_target]))
    u_max = float(np.nanmax([np.nanmax(arr) for arr in all_pred + all_target]))
    field_norm = colors.Normalize(vmin=u_min, vmax=u_max)

    err_abs = float(np.nanmax([np.nanmax(np.abs(arr)) for arr in all_error]))
    err_norm = colors.Normalize(vmin=-err_abs, vmax=err_abs)

    plot_max_prediction_vs_target(
        max_data=max_data,
        plot_dir=plot_dir,
    )

    if plot_animations_rollout:
        anim = animate_sphere_rows(
            rows=rows,
            out_path=os.path.join(
                anim_dir,
                f"training_size_comparison_rollout_idx{rolloutidx}.mp4",
            ),
            fps=10,
            interval=100,
            pred_target_cmap="viridis",
            error_cmap="coolwarm",
            pred_target_norm=field_norm,
            error_norm=None,
            titles=("Prediction", "Target", "Error"),
            colorbar_label="u",
            azim=azim,
            elev=elev,
        )

        return anim


def plot_max_prediction_vs_target(
    max_data,
    plot_dir,
    filename="max_prediction_vs_target_all_training_sizes.png",
):
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, rollout_steps, max_pred, max_target in max_data:
        ax.plot(
            rollout_steps,
            max_pred,
            linestyle="--",
            marker="o",
            label=f"Prediction, {label}",
        )

        ax.plot(
            rollout_steps,
            max_target,
            linestyle="-",
            marker="o",
            label=f"Target, {label}",
        )

    ax.set_xlabel("Rollout", fontsize=14)
    ax.set_ylabel("Maximum value", fontsize=14)
    ax.set_title("Maximum prediction vs target over rollout", fontsize=16)
    ax.grid()
    ax.legend(fontsize=10, ncol=2)

    fig.tight_layout()

    fig.savefig(
        os.path.join(plot_dir, filename),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def animate_sphere_rows(
    rows,
    out_path=None,
    fps=10,
    interval=100,
    pred_target_cmap="viridis",
    error_cmap="coolwarm",
    pred_target_norm=None,
    error_norm=None,
    titles=("Prediction", "Target", "Error"),
    colorbar_label="u",
    elev=20,
    azim=160,
):
    n_rows = len(rows)

    ds_pred0, _, _, _ = rows[0]

    P = ds_pred0["P"].values
    P_plot = P.T if P.shape[0] == 3 and P.shape[1] != 3 else P

    tri = np.asarray(ds_pred0["tri"].values, dtype=int)
    R = float(ds_pred0.attrs["R"])

    tri_vertices = P_plot[tri]
    time_steps = ds_pred0["time"].values

    row_data = []
    all_pred_target = []
    all_error = []

    for ds_pred, ds_target, ds_error, row_label in rows:
        u_pred = ds_pred["u"].values
        u_target = ds_target["u"].values
        u_error = ds_error["u"].values

        if not (u_pred.shape == u_target.shape == u_error.shape):
            raise ValueError(f"Shape mismatch in row: {row_label}")

        if len(ds_pred["time"].values) != len(time_steps):
            raise ValueError(f"Time mismatch in row: {row_label}")

        row_data.append((u_pred, u_target, u_error, row_label))
        all_pred_target.extend([u_pred, u_target])
        all_error.append(u_error)

    if pred_target_norm is None:
        u_min = float(np.nanmin([np.nanmin(arr) for arr in all_pred_target]))
        u_max = float(np.nanmax([np.nanmax(arr) for arr in all_pred_target]))
        pred_target_norm = colors.Normalize(vmin=u_min, vmax=u_max)

    # One fixed-in-time error norm per row
    row_error_norms = []
    for u_error in all_error:
        err_abs = float(np.nanmax(np.abs(u_error)))
        row_error_norms.append(colors.Normalize(vmin=-err_abs, vmax=err_abs))

    cmap_pred_target = cm.get_cmap(pred_target_cmap)
    cmap_error = cm.get_cmap(error_cmap)

    fig = plt.figure(figsize=(12, 4 * n_rows))

    polys = []
    plot_info = []

    for r, (u_pred, u_target, u_error, row_label) in enumerate(row_data):

        error_norm_row = error_norm if error_norm is not None else row_error_norms[r]
        datasets = [
            (u_pred, titles[0], cmap_pred_target, pred_target_norm),
            (u_target, titles[1], cmap_pred_target, pred_target_norm),
            (u_error, titles[2], cmap_error, error_norm_row),
        ]

        for c, (u_data, title, cmap, norm) in enumerate(datasets):
            ax = fig.add_subplot(n_rows, 3, r * 3 + c + 1, projection="3d")

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
            ax.set_zlabel("z", labelpad=-2)

            ax.view_init(elev=elev, azim=azim)

            if r == 0:
                ax.set_title(title, fontsize=13, fontweight="bold",pad=20)

            if c == 0:
                ax.text2D(
                    -0.08,
                    1.05,
                    row_label,
                    transform=ax.transAxes,
                    ha="left",
                    fontsize=11,
                    fontweight="bold",
                )

            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            sm.set_clim(norm.vmin, norm.vmax)

            cbar = fig.colorbar(sm, ax=ax, pad=0.07, shrink=0.75)
            cbar.set_label("error" if title.lower() == "error" else colorbar_label)

            polys.append(poly)
            plot_info.append((u_data, title, cmap, norm, ax))

    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        bottom=0.02,
        top=0.94,
        wspace=0.05,
        hspace=0.05,
    )

    def update(frame):
        current_time = float(time_steps[frame])

        for idx, (poly, (u_data, title, cmap, norm, ax)) in enumerate(zip(polys, plot_info)):
            u = u_data[frame]
            u_face = u[tri].mean(axis=1)

            poly.set_facecolor(cmap(norm(u_face)))
            ax.view_init(elev=elev, azim=azim)

        # Only one global title
        fig.suptitle(
            f"AR rollout: {current_time:.6f}",
            fontsize=16,
            y=0.98,
        )

        return polys

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_steps),
        interval=interval,
        blit=False,
    )

    if out_path:
        writer = "pillow" if out_path.lower().endswith(".gif") else "ffmpeg"

        anim.save(
            out_path,
            writer=writer,
            fps=fps,
            savefig_kwargs={
                "bbox_inches": "tight",
                "pad_inches": 0.05,
            },
        )

    return anim