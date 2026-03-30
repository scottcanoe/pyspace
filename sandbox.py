from pathlib import Path

import numpy as np

from pyspace import Transform, animate_object_poses


def _helix_pose(theta: float, radius: float, pitch: float) -> Transform:
    """Pose whose origin follows a helix, oriented along the tangent."""
    translation = np.array(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            pitch * theta,
        ],
        dtype=float,
    )

    tangent = np.array(
        [-radius * np.sin(theta), radius * np.cos(theta), pitch],
        dtype=float,
    )
    z_axis = tangent / np.linalg.norm(tangent)

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        world_up = np.array([1.0, 0.0, 0.0], dtype=float)
        x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return Transform(rotation=rotation, translation=translation)


def build_helix_poses(
    num_frames: int = 240,
    turns: float = 3.0,
    radius: float = 2.0,
    pitch_per_radian: float = 0.15,
) -> list[Transform]:
    thetas = np.linspace(0.0, 2.0 * np.pi * turns, num_frames)
    return [
        _helix_pose(theta, radius=radius, pitch=pitch_per_radian)
        for theta in thetas
    ]


if __name__ == "__main__":
    output_dir = Path("local/movies")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "helix_reference_frame.gif"

    poses = build_helix_poses()
    animate_object_poses(
        poses,
        interval_ms=1000 // 30,
        axis_scale=0.8,
        show_path=True,
        show=False,
        save_path=str(output_path),
        fps=30,
    )

    print(f"Saved movie to: {output_path.resolve()}")