"""Geometric types that live in reference frames."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from pyspace.protocols import FrameTransformable
from pyspace.transform import FrameTransform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspace.frame import Frame


class Location(FrameTransformable):
    """A point in 3-D space expressed in a particular frame."""

    def __init__(self, array: ArrayLike, frame: Frame) -> None:
        array = np.array(array, dtype=float)
        if array.shape[-1] != 3:
            raise ValueError(
                f"Expected `array` to have shape (..., 3), got {array.shape}."
            )
        self._array = array
        self._frame = frame

    @property
    def frame(self) -> Frame:
        return self._frame

    def as_array(self) -> np.ndarray:
        return self._array.copy()

    @staticmethod
    def from_array(array: ArrayLike, frame: Frame) -> Location:
        return Location(array, frame)

    def apply_frame_transform(self, transform: FrameTransform) -> Location:
        return Location(
            transform.as_rigid_transform().apply(self._array),
            frame=transform.to_frame,
        )

    def __repr__(self) -> str:
        return f"Location({self._array}, frame={self._frame.frame_id})"


class Displacement(FrameTransformable):
    """A free vector — only rotation is applied on transform (translation ignored)."""

    def __init__(self, array: ArrayLike, frame: Frame) -> None:
        array = np.array(array, dtype=float)
        if array.shape[-1] != 3:
            raise ValueError(
                f"Expected `array` to have shape (..., 3), got {array.shape}."
            )
        self._array = array
        self._frame = frame

    @property
    def frame(self) -> Frame:
        return self._frame

    def as_array(self) -> np.ndarray:
        return self._array.copy()

    @staticmethod
    def from_array(array: ArrayLike, frame: Frame) -> Displacement:
        return Displacement(array, frame)

    def apply_frame_transform(self, transform: FrameTransform) -> Displacement:
        return Displacement(
            transform.rotation.apply(self._array),
            frame=transform.to_frame,
        )

    def __repr__(self) -> str:
        return f"Displacement({self._array}, frame={self._frame.frame_id})"


class Orientation(FrameTransformable):
    """A rotation expressed in a particular frame."""

    def __init__(self, rotation: Rotation, frame: Frame) -> None:
        self._rotation = rotation
        self._frame = frame

    @property
    def frame(self) -> Frame:
        return self._frame

    def as_euler(self, dims: str, degrees: bool = False) -> np.ndarray:
        return self._rotation.as_euler(dims, degrees)

    def as_matrix(self) -> np.ndarray:
        return self._rotation.as_matrix()

    def as_rotation(self) -> Rotation:
        return self._rotation

    def as_quat(self) -> np.ndarray:
        return self._rotation.as_quat(scalar_first=True)

    @staticmethod
    def from_euler(
        frame: Frame,
        dims: str,
        angles: ArrayLike,
        degrees: bool = False,
    ) -> Orientation:
        return Orientation(Rotation.from_euler(dims, angles, degrees), frame)

    @staticmethod
    def from_matrix(frame: Frame, matrix: ArrayLike) -> Orientation:
        return Orientation(Rotation.from_matrix(matrix), frame)

    @staticmethod
    def from_rotation(frame: Frame, rotation: Rotation) -> Orientation:
        return Orientation(rotation, frame)

    @staticmethod
    def from_quat(
        frame: Frame,
        quat: ArrayLike,
        scalar_first: bool = True,
    ) -> Orientation:
        return Orientation(
            Rotation.from_quat(quat, scalar_first=scalar_first),
            frame,
        )

    def apply_frame_transform(self, transform: FrameTransform) -> Orientation:
        return Orientation(transform.rotation * self._rotation, transform.to_frame)

    def __repr__(self) -> str:
        return f"Orientation({self._rotation}, frame={self._frame.frame_id})"


class Pose(FrameTransformable):
    """A location and orientation in the same frame."""

    def __init__(self, location: Location, orientation: Orientation) -> None:
        if location.frame != orientation.frame:
            raise ValueError(
                "Pose requires location and orientation in the same frame; "
                f"got {location.frame} and {orientation.frame}"
            )
        self._location = location
        self._orientation = orientation

    @property
    def frame(self) -> Frame:
        return self._location.frame

    @property
    def location(self) -> Location:
        return self._location

    @property
    def orientation(self) -> Orientation:
        return self._orientation

    def apply_frame_transform(self, transform: FrameTransform) -> Pose:
        return Pose(
            location=self._location.apply_frame_transform(transform),
            orientation=self._orientation.apply_frame_transform(transform),
        )

    def __repr__(self) -> str:
        return f"Pose(location={self.location}, orientation={self.orientation})"
