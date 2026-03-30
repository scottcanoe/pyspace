import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import Transform, interpolate_transforms


def test_interpolate_transforms_preserves_endpoints() -> None:
    start = Transform(rotation=np.eye(3), translation=np.array([0.0, 0.0, 0.0]))
    end = Transform(
        rotation=Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        translation=np.array([1.0, 2.0, 3.0]),
    )

    poses = interpolate_transforms([start, end], frames_per_segment=5)
    assert len(poses) == 6
    assert np.allclose(poses[0].translation, start.translation)
    assert np.allclose(poses[-1].translation, end.translation)
    assert np.allclose(poses[-1].rotation, end.rotation)


def test_interpolate_transforms_linear_translation() -> None:
    start = Transform(rotation=np.eye(3), translation=np.array([0.0, 0.0, 0.0]))
    end = Transform(rotation=np.eye(3), translation=np.array([2.0, 0.0, 0.0]))

    poses = interpolate_transforms([start, end], frames_per_segment=4)
    expected_x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    actual_x = np.array([pose.translation[0] for pose in poses])
    assert np.allclose(actual_x, expected_x)
