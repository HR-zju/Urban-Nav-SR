"""Global ENU BEV visualization shared by the Orin NavFlow runners.

The layout and target-state semantics mirror the BEV output in go2's
``model_inference.py``: a full-history global view plus a 20 m reset window.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def update_bev_window(current_pose, view_range, window_size_m=20.0):
    """Keep a fixed ENU window and reset it after the robot leaves it."""
    robot_x = float(current_pose[0])
    robot_y = float(current_pose[1])
    needs_reset = view_range is None

    if view_range is not None:
        view_range_x, view_range_y = view_range
        needs_reset = not (
            view_range_x[0] <= robot_x <= view_range_x[1]
            and view_range_y[0] <= robot_y <= view_range_y[1]
        )

    if needs_reset:
        half_size = float(window_size_m) / 2.0
        view_range = (
            (robot_x - half_size, robot_x + half_size),
            (robot_y - half_size, robot_y + half_size),
        )

    return view_range, needs_reset


def select_targets_for_bev(
    all_targets,
    remaining_target_count,
    current_pose,
    display_range_m=80.0,
):
    """Return nearby targets with reached/active/pending states."""
    active_target_index = len(all_targets) - int(remaining_target_count)
    robot_xy = np.asarray(current_pose[:2], dtype=float)
    target_points = []
    target_states = []
    target_indices = []

    for target_index, target_pose in enumerate(all_targets):
        target_point = np.asarray(target_pose[:2], dtype=float)
        if target_index < active_target_index:
            target_state = "reached"
        elif target_index == active_target_index:
            target_state = "active"
        else:
            target_state = "pending"

        target_distance = float(np.linalg.norm(target_point - robot_xy))
        if target_distance <= float(display_range_m) or target_state == "active":
            target_points.append(target_point)
            target_states.append(target_state)
            target_indices.append(target_index)

    return (
        np.asarray(target_points, dtype=float).reshape(-1, 2),
        target_states,
        target_indices,
    )


def project_waypoints_to_bev_image_enu(
    waypoints,
    post_odom=None,
    targets_enu=None,
    target_states=None,
    target_indices=None,
    current_pose=None,
    save_path=None,
    title=None,
    figsize=(4.8, 6.4),
    dpi=100,
    view_range_x=(-10, 10),
    view_range_y=(-10, 10),
):
    """Draw full ENU history, target states, prediction, and reset window."""
    fig = None
    try:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)

        history_enu = None
        if post_odom is not None:
            history_enu = np.asarray(post_odom, dtype=float).reshape(-1, 2)
            if len(history_enu) == 0:
                history_enu = None

        if current_pose is None or len(current_pose) < 7:
            raise ValueError("current_pose must be [x, y, z, qx, qy, qz, qw]")
        robot_x = float(current_pose[0])
        robot_y = float(current_pose[1])

        plotted_targets = np.empty((0, 2), dtype=float)
        plotted_states = []
        plotted_indices = []
        if targets_enu is not None:
            target_array = np.asarray(targets_enu, dtype=float)
            if target_array.size >= 2:
                plotted_targets = target_array.reshape(
                    -1, target_array.shape[-1]
                )[:, :2]
                plotted_states = list(target_states or [])
                plotted_indices = list(target_indices or [])
                target_count = min(
                    len(plotted_targets),
                    len(plotted_states),
                    len(plotted_indices),
                )
                plotted_targets = plotted_targets[:target_count]
                plotted_states = plotted_states[:target_count]
                plotted_indices = plotted_indices[:target_count]

        active_target = None
        for target_point, target_state in zip(
            plotted_targets, plotted_states
        ):
            if target_state == "active":
                active_target = target_point
                break

        waypoints_enu = None
        if waypoints is not None and len(waypoints) > 0:
            waypoints_local = np.asarray(waypoints, dtype=float)
            waypoints_xyz = np.zeros((len(waypoints_local), 3), dtype=float)
            waypoints_xyz[:, :2] = waypoints_local[:, :2]
            rotation_enu_from_base = R.from_quat(current_pose[3:7])
            waypoints_enu = rotation_enu_from_base.apply(waypoints_xyz)
            waypoints_enu += np.asarray(current_pose[:3], dtype=float)

        if history_enu is not None:
            ax.plot(
                history_enu[:, 0],
                history_enu[:, 1],
                "-k",
                linewidth=1.8,
                label="all history",
            )

        if waypoints_enu is not None:
            ax.plot(
                waypoints_enu[:, 0],
                waypoints_enu[:, 1],
                "-b",
                marker="o",
                label="pred",
            )
            ax.scatter(
                waypoints_enu[0, 0],
                waypoints_enu[0, 1],
                c="cyan",
                s=30,
            )

        target_style = {
            "reached": {
                "color": "forestgreen",
                "marker": "X",
                "size": 52,
                "label": "reached target",
            },
            "active": {
                "color": "red",
                "marker": "*",
                "size": 125,
                "label": "active target",
            },
            "pending": {
                "color": "darkorange",
                "marker": "o",
                "size": 50,
                "label": "pending target",
            },
        }
        labeled_states = set()
        for target_point, target_state, target_index in zip(
            plotted_targets,
            plotted_states,
            plotted_indices,
        ):
            style = target_style[target_state]
            label = (
                style["label"] if target_state not in labeled_states else None
            )
            scatter_kwargs = {
                "marker": style["marker"],
                "s": style["size"],
                "label": label,
                "zorder": 12 if target_state == "active" else 11,
                "linewidths": 1.5,
            }
            if target_state == "pending":
                scatter_kwargs.update(
                    facecolors="none",
                    edgecolors=style["color"],
                )
            else:
                scatter_kwargs.update(c=style["color"])
            ax.scatter(target_point[0], target_point[1], **scatter_kwargs)
            labeled_states.add(target_state)

            target_name = f"T{target_index + 1}"
            if target_state == "active":
                target_name += (
                    f" ({target_point[0]:.1f}, {target_point[1]:.1f})"
                )
            ax.annotate(
                target_name,
                target_point,
                xytext=(5, 5),
                textcoords="offset points",
                color=style["color"],
                fontsize=7,
            )

        ax.scatter(
            robot_x,
            robot_y,
            c="green",
            s=20,
            label="robot",
            zorder=10,
            edgecolors="white",
            linewidths=1.5,
        )
        if title is not None:
            ax.set_title(title, fontsize=12)
        ax.set_xlabel("E (m)")
        ax.set_ylabel("N (m)")
        ax.legend(loc="upper left", fontsize=7)

        extent_points = [np.array([[robot_x, robot_y]])]
        if history_enu is not None:
            extent_points.append(history_enu)
        if waypoints_enu is not None:
            extent_points.append(waypoints_enu[:, :2])
        if len(plotted_targets) > 0:
            extent_points.append(plotted_targets)
        extent = np.concatenate(extent_points, axis=0)
        min_x, min_y = np.min(extent, axis=0)
        max_x, max_y = np.max(extent, axis=0)
        span_x = max(max_x - min_x, 20.0)
        span_y = max(max_y - min_y, 20.0)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        padding_x = max(2.0, span_x * 0.08)
        padding_y = max(2.0, span_y * 0.08)
        ax.set_xlim(
            center_x - span_x / 2.0 - padding_x,
            center_x + span_x / 2.0 + padding_x,
        )
        ax.set_ylim(
            center_y - span_y / 2.0 - padding_y,
            center_y + span_y / 2.0 + padding_y,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.5)

        local_ax = ax.inset_axes([0.58, 0.05, 0.39, 0.39])
        local_ax.set_facecolor((1.0, 1.0, 1.0, 0.92))
        if history_enu is not None:
            local_ax.plot(
                history_enu[:, 0],
                history_enu[:, 1],
                "-k",
                linewidth=1.2,
            )
        if waypoints_enu is not None:
            local_ax.plot(
                waypoints_enu[:, 0],
                waypoints_enu[:, 1],
                "-b",
                marker="o",
                markersize=2.5,
            )
        local_ax.scatter(robot_x, robot_y, c="green", s=18, zorder=10)

        for target_point, target_state in zip(
            plotted_targets, plotted_states
        ):
            target_x, target_y = target_point
            if (
                view_range_x[0] <= target_x <= view_range_x[1]
                and view_range_y[0] <= target_y <= view_range_y[1]
            ):
                style = target_style[target_state]
                local_kwargs = {
                    "marker": style["marker"],
                    "s": 45 if target_state == "active" else 28,
                    "zorder": 11,
                    "linewidths": 1.1,
                }
                if target_state == "pending":
                    local_kwargs.update(
                        facecolors="none",
                        edgecolors=style["color"],
                    )
                else:
                    local_kwargs.update(c=style["color"])
                local_ax.scatter(target_x, target_y, **local_kwargs)

        if active_target is not None:
            target_x, target_y = active_target
            in_local_window = (
                view_range_x[0] <= target_x <= view_range_x[1]
                and view_range_y[0] <= target_y <= view_range_y[1]
            )
            if not in_local_window:
                ray_x = target_x - robot_x
                ray_y = target_y - robot_y
                intersections = []
                if abs(ray_x) > 1e-9:
                    for boundary_x in view_range_x:
                        scale = (boundary_x - robot_x) / ray_x
                        candidate_y = robot_y + scale * ray_y
                        if (
                            scale >= 0.0
                            and view_range_y[0]
                            <= candidate_y
                            <= view_range_y[1]
                        ):
                            intersections.append(
                                (scale, boundary_x, candidate_y)
                            )
                if abs(ray_y) > 1e-9:
                    for boundary_y in view_range_y:
                        scale = (boundary_y - robot_y) / ray_y
                        candidate_x = robot_x + scale * ray_x
                        if (
                            scale >= 0.0
                            and view_range_x[0]
                            <= candidate_x
                            <= view_range_x[1]
                        ):
                            intersections.append(
                                (scale, candidate_x, boundary_y)
                            )
                if intersections:
                    _, edge_x, edge_y = min(
                        intersections, key=lambda item: item[0]
                    )
                    local_ax.scatter(
                        edge_x,
                        edge_y,
                        c="red",
                        marker="v",
                        s=45,
                        zorder=11,
                    )

        local_ax.set_xlim(view_range_x)
        local_ax.set_ylim(view_range_y)
        local_ax.set_aspect("equal", adjustable="box")
        local_ax.set_title("20 m reset window", fontsize=7)
        local_ax.set_xlabel("E", fontsize=6)
        local_ax.set_ylabel("N", fontsize=6)
        local_ax.tick_params(axis="both", labelsize=5)
        local_ax.grid(True, linestyle="--", alpha=0.4)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
            return save_path
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"BEV plot failed: {exc}")
        return None
    finally:
        if fig is not None:
            plt.close(fig)
