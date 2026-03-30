"""First-class reference frames, transforms, and frame-bound coordinates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, Self, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


class FrameGraphError(Exception):
    """Base exception for frame graph operations."""


class FrameNotFoundError(FrameGraphError):
    """Raised when a frame is not found in the graph."""


class TransformNotFoundError(FrameGraphError):
    """Raised when no transform path exists between frames."""


class FrameMismatchError(FrameGraphError):
    """Raised when applying a transform to an object in a different frame."""


class Frame:
    """Reference frame object managed by a `FrameGraph`."""

    def __init__(self, graph: FrameGraph, name: str) -> None:
        self._graph = graph
        self._name = name

    @property
    def graph(self) -> FrameGraph:
        return self._graph

    @property
    def name(self) -> str:
        return self._name

    def location(self, coordinates: ArrayLike) -> Location:
        return Location(coordinates, frame=self)

    def orientation(self, rotation: Rotation) -> Orientation:
        return Orientation(rotation, frame=self)

    def pose(self, translation: ArrayLike, rotation: Rotation) -> Pose:
        return Pose(
            location=Location(translation, frame=self),
            orientation=Orientation(rotation, frame=self),
        )

    def __repr__(self) -> str:
        return f"Frame({self._name})"


@dataclass(frozen=True, slots=True)
class FrameTransform:
    """Rigid transform from `from_frame` coordinates into `to_frame` coordinates."""

    rotation: np.ndarray
    translation: np.ndarray
    from_frame: Frame
    to_frame: Frame

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must be (3, 3), got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"translation must be (3,), got {translation.shape}")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @staticmethod
    def identity(frame: Frame) -> FrameTransform:
        return FrameTransform(
            rotation=np.eye(3),
            translation=np.zeros(3),
            from_frame=frame,
            to_frame=frame,
        )

    @staticmethod
    def from_components(
        *,
        translation: ArrayLike,
        rotation: Rotation,
        from_frame: Frame,
        to_frame: Frame,
    ) -> FrameTransform:
        return FrameTransform(
            rotation=rotation.as_matrix(),
            translation=np.asarray(translation, dtype=float),
            from_frame=from_frame,
            to_frame=to_frame,
        )

    @property
    def rotation_object(self) -> Rotation:
        return Rotation.from_matrix(self.rotation)

    def inverse(self) -> FrameTransform:
        inv_rotation = self.rotation.T
        inv_translation = -(inv_rotation @ self.translation)
        return FrameTransform(
            rotation=inv_rotation,
            translation=inv_translation,
            from_frame=self.to_frame,
            to_frame=self.from_frame,
        )

    def compose(self, other: FrameTransform) -> FrameTransform:
        """Return `self(other(x))` with strict frame compatibility checks."""
        if other.to_frame != self.from_frame:
            raise FrameMismatchError(
                f"cannot compose {self} and {other}: frame mismatch "
                f"{other.to_frame} != {self.from_frame}"
            )
        composed_rotation = self.rotation @ other.rotation
        composed_translation = self.rotation @ other.translation + self.translation
        return FrameTransform(
            rotation=composed_rotation,
            translation=composed_translation,
            from_frame=other.from_frame,
            to_frame=self.to_frame,
        )

    def apply_coordinates(self, coordinates: ArrayLike) -> np.ndarray:
        points = np.asarray(coordinates, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError(
                f"coordinates must end in size 3 dimension, got {points.shape}"
            )
        return points @ self.rotation.T + self.translation

    def apply(self, obj: TFrameTransformable) -> TFrameTransformable:
        if obj.frame != self.from_frame:
            raise FrameMismatchError(
                f"object is in {obj.frame}, expected {self.from_frame}"
            )
        result = obj.apply_frame_transform(self)
        if result.frame != self.to_frame:
            raise FrameMismatchError(
                f"transformed object in {result.frame}, expected {self.to_frame}"
            )
        return result

    def __repr__(self) -> str:
        return f"FrameTransform({self.from_frame.name} -> {self.to_frame.name})"


@dataclass(frozen=True, slots=True)
class PathStep:
    transform: FrameTransform
    inverse: bool


class FrameTransformable(Protocol):
    @property
    def frame(self) -> Frame: ...

    def apply_frame_transform(self, transform: FrameTransform) -> Self: ...

    def to(self, frame: Frame | str) -> Self: ...


TFrameTransformable = TypeVar("TFrameTransformable", bound=FrameTransformable)


class Location(FrameTransformable):
    def __init__(self, coordinates: ArrayLike, frame: Frame) -> None:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape[-1] != 3:
            raise ValueError(
                f"coordinates must end in size 3 dimension, got {coordinates.shape}"
            )
        self._coordinates = coordinates
        self._frame = frame

    @property
    def frame(self) -> Frame:
        return self._frame

    def as_array(self) -> np.ndarray:
        return self._coordinates.copy()

    def apply_frame_transform(self, transform: FrameTransform) -> Location:
        return Location(transform.apply_coordinates(self._coordinates), transform.to_frame)

    def to(self, frame: Frame | str) -> Location:
        return self._frame.graph.transform(self, frame)

    def __repr__(self) -> str:
        return f"Location({self._coordinates}, frame={self._frame.name})"


class Orientation(FrameTransformable):
    def __init__(self, rotation: Rotation, frame: Frame) -> None:
        self._rotation = rotation
        self._frame = frame

    @property
    def frame(self) -> Frame:
        return self._frame

    def as_rotation(self) -> Rotation:
        return self._rotation

    def as_matrix(self) -> np.ndarray:
        return self._rotation.as_matrix()

    def apply_frame_transform(self, transform: FrameTransform) -> Orientation:
        rotated = transform.rotation_object * self._rotation
        return Orientation(rotated, transform.to_frame)

    def to(self, frame: Frame | str) -> Orientation:
        return self._frame.graph.transform(self, frame)

    def __repr__(self) -> str:
        return f"Orientation({self._rotation}, frame={self._frame.name})"


class Pose(FrameTransformable):
    def __init__(self, location: Location, orientation: Orientation) -> None:
        if location.frame != orientation.frame:
            raise ValueError("location and orientation must share the same frame")
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

    def to(self, frame: Frame | str) -> Pose:
        return self.frame.graph.transform(self, frame)

    def __repr__(self) -> str:
        return f"Pose(location={self._location}, orientation={self._orientation})"


class FrameGraph:
    """Graph of first-class frames connected by first-class transforms."""

    def __init__(self) -> None:
        self._frames_by_name: dict[str, Frame] = {}
        self._transforms_by_pair: dict[tuple[Frame, Frame], FrameTransform] = {}
        self._adjacency: dict[Frame, dict[Frame, PathStep]] = {}

    @property
    def frames(self) -> tuple[Frame, ...]:
        return tuple(sorted(self._frames_by_name.values(), key=lambda frame: frame.name))

    @property
    def transforms(self) -> tuple[FrameTransform, ...]:
        return tuple(self._transforms_by_pair.values())

    def add_frame(self, name: str) -> Frame:
        if name in self._frames_by_name:
            raise FrameGraphError(f"frame '{name}' already exists")
        frame = Frame(graph=self, name=name)
        self._frames_by_name[name] = frame
        self._adjacency[frame] = {}
        return frame

    def add_frames(self, names: list[str]) -> tuple[Frame, ...]:
        return tuple(self.add_frame(name) for name in names)

    def add_transform(self, transform: FrameTransform) -> FrameTransform:
        self._require_graph_frame(transform.from_frame)
        self._require_graph_frame(transform.to_frame)
        if transform.from_frame == transform.to_frame:
            raise FrameGraphError("cannot add transform from frame to itself")
        forward_pair = (transform.from_frame, transform.to_frame)
        reverse_pair = (transform.to_frame, transform.from_frame)
        if forward_pair in self._transforms_by_pair or reverse_pair in self._transforms_by_pair:
            raise FrameGraphError(
                f"transform already exists between {transform.from_frame} and {transform.to_frame}"
            )
        self._transforms_by_pair[forward_pair] = transform
        self._adjacency[transform.from_frame][transform.to_frame] = PathStep(
            transform=transform,
            inverse=False,
        )
        self._adjacency[transform.to_frame][transform.from_frame] = PathStep(
            transform=transform,
            inverse=True,
        )
        return transform

    def get_frame(self, frame: Frame | str) -> Frame:
        if isinstance(frame, Frame):
            self._require_graph_frame(frame)
            return frame
        try:
            return self._frames_by_name[frame]
        except KeyError as exc:
            raise FrameNotFoundError(f"unknown frame '{frame}'") from exc

    def shortest_path(self, from_frame: Frame | str, to_frame: Frame | str) -> list[PathStep]:
        source = self.get_frame(from_frame)
        target = self.get_frame(to_frame)
        if source == target:
            return []

        queue: deque[Frame] = deque([source])
        previous: dict[Frame, tuple[Frame | None, PathStep | None]] = {
            source: (None, None)
        }
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for neighbor, step in self._adjacency[current].items():
                if neighbor not in previous:
                    previous[neighbor] = (current, step)
                    queue.append(neighbor)

        if target not in previous:
            raise TransformNotFoundError(f"no path from {source} to {target}")

        reversed_path: list[PathStep] = []
        current = target
        while True:
            parent, step = previous[current]
            if parent is None or step is None:
                break
            reversed_path.append(step)
            current = parent
        reversed_path.reverse()
        return reversed_path

    def get_transform(self, from_frame: Frame | str, to_frame: Frame | str) -> FrameTransform:
        source = self.get_frame(from_frame)
        target = self.get_frame(to_frame)
        if source == target:
            return FrameTransform.identity(source)

        path = self.shortest_path(source, target)
        composed = FrameTransform.identity(source)
        for step in path:
            transform = step.transform.inverse() if step.inverse else step.transform
            composed = transform.compose(composed)
        return composed

    def transform(
        self,
        obj: TFrameTransformable,
        to_frame: Frame | str,
    ) -> TFrameTransformable:
        path = self.shortest_path(obj.frame, to_frame)
        transformed = obj
        for step in path:
            transform = step.transform.inverse() if step.inverse else step.transform
            transformed = transform.apply(transformed)
        return transformed

    def transform_coordinates(
        self,
        coordinates: ArrayLike,
        *,
        from_frame: Frame | str,
        to_frame: Frame | str,
    ) -> np.ndarray:
        transform = self.get_transform(from_frame, to_frame)
        return transform.apply_coordinates(coordinates)

    def _require_graph_frame(self, frame: Frame) -> None:
        known = self._frames_by_name.get(frame.name)
        if known is None or known is not frame:
            raise FrameNotFoundError(f"frame '{frame}' is not in this graph")


# Backward-compatible alias.
Transform = FrameTransform
