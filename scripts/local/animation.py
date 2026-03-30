"""Utilities for animating rigid object poses in 3D."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation, Slerp

from pyspace.geometry import Location, Orientation, Pose

DEFAULT_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
    ],
    dtype=float,
)

DEFAULT_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def interpolate_poses(
    keyframes: Sequence[Pose],
    *,
    frames_per_segment: int = 20,
) -> list[Pose]:
    """Interpolate a smooth pose trajectory through transform keyframes."""
    if len(keyframes) < 2:
        raise ValueError("keyframes must contain at least two transforms")
    if frames_per_segment < 2:
        raise ValueError("frames_per_segment must be >= 2")

    trajectory: list[Pose] = []
    for index in range(len(keyframes) - 1):
        start = keyframes[index]
        end = keyframes[index + 1]
        if start.frame != end.frame:
            raise ValueError("all keyframes must be in the same frame")

        key_times = np.array([0.0, 1.0], dtype=float)
        key_rots = Rotation.from_matrix(
            np.stack(
                [
                    start.orientation.as_rotation().as_matrix(),
                    end.orientation.as_rotation().as_matrix(),
                ],
                axis=0,
            )
        )
        slerp = Slerp(key_times, key_rots)
        samples = np.linspace(0.0, 1.0, frames_per_segment, endpoint=False)
        for t in samples:
            translation = (1.0 - t) * start.location.as_array() + t * end.location.as_array()
            rotation = slerp([t])[0]
            trajectory.append(
                Pose(
                    location=Location(translation, frame=start.frame),
                    orientation=Orientation(rotation, frame=start.frame),
                )
            )

    trajectory.append(keyframes[-1])
    return trajectory


def animate_object_poses(
    poses: Sequence[Pose],
    *,
    object_vertices: ArrayLike | None = None,
    object_edges: Sequence[tuple[int, int]] = DEFAULT_EDGES,
    interval_ms: int = 33,
    axis_scale: float = 0.6,
    show_path: bool = True,
    show: bool = True,
    save_path: str | None = None,
    fps: int = 30,
) -> tuple[Any, Any]:
    """Animate an object moving through space with translation and rotation."""
    if len(poses) == 0:
        raise ValueError("poses must contain at least one Pose")
    if len({pose.frame for pose in poses}) != 1:
        raise ValueError("all poses must be expressed in the same frame")

    vertices = np.asarray(
        DEFAULT_VERTICES if object_vertices is None else object_vertices,
        dtype=float,
    )
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"object_vertices must have shape (N, 3), got {vertices.shape}"
        )

    transformed_vertices = np.stack([pose.location.as_array() for pose in poses], axis=0)
    transformed_vertices = np.stack(
        [
            pose.location.as_array()
            + vertices @ pose.orientation.as_matrix().T
            for pose in poses
        ],
        axis=0,
    )
    centers = np.stack([pose.location.as_array() for pose in poses], axis=0)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as exc:  # pragma: no cover - import-time dependency issue
        raise RuntimeError(
            "matplotlib is required for animation. Install it to use this feature."
        ) from exc

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Object pose animation")

    # Keep equal-ish limits so rotation remains visually meaningful.
    all_points = transformed_vertices.reshape(-1, 3)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.55, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    edge_artists = [ax.plot([], [], [], color="black", lw=2)[0] for _ in object_edges]
    axis_colors = ("red", "green", "blue")
    axis_artists = [ax.plot([], [], [], color=c, lw=2)[0] for c in axis_colors]

    path_artist = ax.plot([], [], [], color="purple", lw=1.5, alpha=0.6)[0]

    def _set_line_3d(artist: Any, p0: np.ndarray, p1: np.ndarray) -> None:
        artist.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        artist.set_3d_properties([p0[2], p1[2]])

    def update(frame_idx: int):
        verts = transformed_vertices[frame_idx]
        pose = poses[frame_idx]

        for artist, (i, j) in zip(edge_artists, object_edges):
            _set_line_3d(artist, verts[i], verts[j])

        origin = pose.location.as_array()
        rotated_basis = pose.orientation.as_matrix() @ np.eye(3)
        for artist, axis_vec in zip(axis_artists, rotated_basis.T):
            _set_line_3d(artist, origin, origin + axis_scale * axis_vec)

        if show_path:
            path = centers[: frame_idx + 1]
            path_artist.set_data(path[:, 0], path[:, 1])
            path_artist.set_3d_properties(path[:, 2])
        else:
            path_artist.set_data([], [])
            path_artist.set_3d_properties([])

        return (*edge_artists, *axis_artists, path_artist)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(poses),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path is not None:
        animation.save(save_path, fps=fps)
    if show:
        plt.show()
    return fig, animation


# Backward-compatible alias for the previous name.
interpolate_transforms = interpolate_poses
