import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import FrameGraph, interpolate_poses


def test_interpolate_poses_preserves_endpoints() -> None:
    graph = FrameGraph()
    world = graph.add_frame("world")
    start = world.pose(np.array([0.0, 0.0, 0.0]), Rotation.identity())
    end = world.pose(np.array([1.0, 2.0, 3.0]), Rotation.from_euler("z", 90, degrees=True))

    poses = interpolate_poses([start, end], frames_per_segment=5)
    assert len(poses) == 6
    assert np.allclose(poses[0].location.as_array(), start.location.as_array())
    assert np.allclose(poses[-1].location.as_array(), end.location.as_array())
    assert np.allclose(
        poses[-1].orientation.as_matrix(),
        end.orientation.as_matrix(),
    )


def test_interpolate_poses_linear_translation() -> None:
    graph = FrameGraph()
    world = graph.add_frame("world")
    start = world.pose(np.array([0.0, 0.0, 0.0]), Rotation.identity())
    end = world.pose(np.array([2.0, 0.0, 0.0]), Rotation.identity())

    poses = interpolate_poses([start, end], frames_per_segment=4)
    expected_x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    actual_x = np.array([pose.location.as_array()[0] for pose in poses])
    assert np.allclose(actual_x, expected_x)
