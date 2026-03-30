import numpy as np
import pytest

from pyspace import FrameGraph, FrameNotFoundError, Transform, TransformNotFoundError


def test_transform_coordinates_direct() -> None:
    graph = FrameGraph()
    graph.add_frames(["world", "camera"])
    graph.add_transform(
        from_frame="camera",
        to_frame="world",
        transform=Transform(rotation=np.eye(3), translation=np.array([1.0, 2.0, 3.0])),
    )

    camera_point = np.array([1.0, 1.0, 1.0])
    world_point = graph.transform_coordinates(
        camera_point,
        from_frame="camera",
        to_frame="world",
    )
    assert np.allclose(world_point, np.array([2.0, 3.0, 4.0]))


def test_transform_coordinates_across_multiple_frames() -> None:
    graph = FrameGraph()
    graph.add_frames(["c", "b", "a"])

    graph.add_transform(
        from_frame="c",
        to_frame="b",
        transform=Transform(rotation=np.eye(3), translation=np.array([0.0, 0.0, 1.0])),
    )
    graph.add_transform(
        from_frame="b",
        to_frame="a",
        transform=Transform(rotation=np.eye(3), translation=np.array([0.0, 2.0, 0.0])),
    )

    point_in_c = np.array([1.0, 1.0, 1.0])
    point_in_a = graph.transform_coordinates(
        point_in_c,
        from_frame="c",
        to_frame="a",
    )
    assert np.allclose(point_in_a, np.array([1.0, 3.0, 2.0]))


def test_missing_frame_raises() -> None:
    graph = FrameGraph()
    graph.add_frame("a")
    with pytest.raises(FrameNotFoundError):
        graph.get_transform("a", "missing")


def test_disconnected_frames_raise() -> None:
    graph = FrameGraph()
    graph.add_frames(["a", "b", "c"])
    graph.add_transform(
        from_frame="a",
        to_frame="b",
        transform=Transform.identity(),
    )
    with pytest.raises(TransformNotFoundError):
        graph.get_transform("a", "c")
