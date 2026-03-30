import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyspace import (
    FrameGraph,
    FrameNotFoundError,
    FrameTransform,
    TransformNotFoundError,
)


def test_transform_coordinates_direct() -> None:
    graph = FrameGraph()
    world, camera = graph.add_frames(["world", "camera"])
    graph.add_transform(
        FrameTransform.from_components(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation=Rotation.identity(),
            from_frame=camera,
            to_frame=world,
        )
    )

    camera_point = np.array([1.0, 1.0, 1.0])
    world_point = graph.transform_coordinates(
        camera_point,
        from_frame=camera,
        to_frame=world,
    )
    assert np.allclose(world_point, np.array([2.0, 3.0, 4.0]))


def test_transform_coordinates_across_multiple_frames() -> None:
    graph = FrameGraph()
    c, b, a = graph.add_frames(["c", "b", "a"])

    graph.add_transform(
        FrameTransform.from_components(
            translation=np.array([0.0, 0.0, 1.0]),
            rotation=Rotation.identity(),
            from_frame=c,
            to_frame=b,
        )
    )
    graph.add_transform(
        FrameTransform.from_components(
            translation=np.array([0.0, 2.0, 0.0]),
            rotation=Rotation.identity(),
            from_frame=b,
            to_frame=a,
        )
    )

    point_in_c = np.array([1.0, 1.0, 1.0])
    point_in_a = graph.transform_coordinates(
        point_in_c,
        from_frame=c,
        to_frame=a,
    )
    assert np.allclose(point_in_a, np.array([1.0, 3.0, 2.0]))


def test_missing_frame_raises() -> None:
    graph = FrameGraph()
    a = graph.add_frame("a")
    with pytest.raises(FrameNotFoundError):
        graph.get_transform(a, "missing")


def test_disconnected_frames_raise() -> None:
    graph = FrameGraph()
    a, b, c = graph.add_frames(["a", "b", "c"])
    graph.add_transform(
        FrameTransform.from_components(
            translation=np.zeros(3),
            rotation=Rotation.identity(),
            from_frame=a,
            to_frame=b,
        )
    )
    with pytest.raises(TransformNotFoundError):
        graph.get_transform(a, c)


def test_location_to_frame_uses_first_class_transform_path() -> None:
    graph = FrameGraph()
    c, b, a = graph.add_frames(["c", "b", "a"])
    graph.add_transform(
        FrameTransform.from_components(
            translation=np.array([1.0, 0.0, 0.0]),
            rotation=Rotation.identity(),
            from_frame=c,
            to_frame=b,
        )
    )
    graph.add_transform(
        FrameTransform.from_components(
            translation=np.array([0.0, 1.0, 0.0]),
            rotation=Rotation.identity(),
            from_frame=b,
            to_frame=a,
        )
    )

    point_c = c.location([1.0, 2.0, 3.0])
    point_a = point_c.to(a)
    assert point_a.frame == a
    assert np.allclose(point_a.as_array(), np.array([2.0, 3.0, 3.0]))
