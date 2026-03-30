"""Reference frame graph and on-demand coordinate transforms."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike


class FrameGraphError(Exception):
    """Base exception for frame graph operations."""


class FrameNotFoundError(FrameGraphError):
    """Raised when a frame name does not exist in the graph."""


class TransformNotFoundError(FrameGraphError):
    """Raised when no path exists between two frames."""


@dataclass(frozen=True, slots=True)
class Transform:
    """Rigid transform from one frame coordinates to another."""

    rotation: np.ndarray
    translation: np.ndarray

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
    def identity() -> Transform:
        return Transform(rotation=np.eye(3), translation=np.zeros(3))

    def inverse(self) -> Transform:
        inv_rotation = self.rotation.T
        inv_translation = -(inv_rotation @ self.translation)
        return Transform(rotation=inv_rotation, translation=inv_translation)

    def compose(self, other: Transform) -> Transform:
        """Compose two transforms as self(other(x))."""
        rotation = self.rotation @ other.rotation
        translation = self.rotation @ other.translation + self.translation
        return Transform(rotation=rotation, translation=translation)

    def apply(self, coordinates: ArrayLike) -> np.ndarray:
        points = np.asarray(coordinates, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError(
                f"coordinates must end in size 3 dimension, got {points.shape}"
            )
        return points @ self.rotation.T + self.translation


class FrameGraph:
    """Undirected frame graph with directed transform edges."""

    def __init__(self) -> None:
        self._frames: set[str] = set()
        self._adjacency: dict[str, dict[str, Transform]] = {}

    @property
    def frames(self) -> tuple[str, ...]:
        return tuple(sorted(self._frames))

    def add_frame(self, name: str) -> str:
        if name in self._frames:
            raise FrameGraphError(f"frame '{name}' already exists")
        self._frames.add(name)
        self._adjacency[name] = {}
        return name

    def add_frames(self, names: Iterable[str]) -> tuple[str, ...]:
        return tuple(self.add_frame(name) for name in names)

    def add_transform(self, from_frame: str, to_frame: str, transform: Transform) -> None:
        self._require_frame(from_frame)
        self._require_frame(to_frame)
        if from_frame == to_frame:
            raise FrameGraphError("cannot set transform for the same frame")
        if to_frame in self._adjacency[from_frame]:
            raise FrameGraphError(
                f"transform already exists between '{from_frame}' and '{to_frame}'"
            )
        self._adjacency[from_frame][to_frame] = transform
        self._adjacency[to_frame][from_frame] = transform.inverse()

    def get_transform(self, from_frame: str, to_frame: str) -> Transform:
        self._require_frame(from_frame)
        self._require_frame(to_frame)
        if from_frame == to_frame:
            return Transform.identity()
        path = self._shortest_path(from_frame, to_frame)
        if not path:
            raise TransformNotFoundError(
                f"no transform path from '{from_frame}' to '{to_frame}'"
            )
        transform = Transform.identity()
        for src, dst in path:
            transform = self._adjacency[src][dst].compose(transform)
        return transform

    def transform_coordinates(
        self,
        coordinates: ArrayLike,
        *,
        from_frame: str,
        to_frame: str,
    ) -> np.ndarray:
        transform = self.get_transform(from_frame=from_frame, to_frame=to_frame)
        return transform.apply(coordinates)

    def _shortest_path(self, from_frame: str, to_frame: str) -> list[tuple[str, str]]:
        if from_frame == to_frame:
            return []
        queue: deque[str] = deque([from_frame])
        predecessor: dict[str, str | None] = {from_frame: None}

        while queue:
            current = queue.popleft()
            if current == to_frame:
                break
            for neighbor in self._adjacency[current]:
                if neighbor not in predecessor:
                    predecessor[neighbor] = current
                    queue.append(neighbor)

        if to_frame not in predecessor:
            return []

        nodes: list[str] = [to_frame]
        while predecessor[nodes[-1]] is not None:
            nodes.append(predecessor[nodes[-1]])  # type: ignore[arg-type]
        nodes.reverse()
        return list(zip(nodes[:-1], nodes[1:]))

    def _require_frame(self, name: str) -> None:
        if name not in self._frames:
            raise FrameNotFoundError(f"unknown frame '{name}'")
