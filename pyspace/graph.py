"""Frame graph: a directed graph of reference frames and transforms."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from bidict import bidict
from scipy.sparse import csgraph

from pyspace.exceptions import GraphError
from pyspace.frame import Frame, FrameID
from pyspace.transform import FrameTransform

if TYPE_CHECKING:
    from pyspace.protocols import TFrameTransformable


@dataclass(frozen=True)
class PathStep:
    transform: FrameTransform
    invert: bool


@dataclass
class PathCache:
    """Cached shortest-path data for the current graph topology."""

    frame_index: bidict[Frame, int]
    directions: np.ndarray
    predecessors: np.ndarray


class FrameGraph:
    """A directed graph of reference frames connected by rigid transforms.

    Transforms between arbitrary frames are found via shortest path on the
    undirected view of the graph, then composed (inverting edges traversed
    backward).
    """

    def __init__(self) -> None:
        self._frames: bidict[FrameID, Frame] = bidict()
        self._transforms: bidict[tuple[Frame, Frame], FrameTransform] = bidict()
        self._path_cache: PathCache | None = None

    @property
    def frames(self) -> bidict[FrameID, Frame]:
        return bidict(self._frames)

    @property
    def transforms(self) -> bidict[tuple[Frame, Frame], FrameTransform]:
        return bidict(self._transforms)

    def add_frame(self, frame_id: FrameID | None = None) -> Frame:
        """Add a frame to the graph."""
        frame_id = FrameID(str(uuid.uuid4())) if frame_id is None else frame_id
        if frame_id in self._frames:
            raise GraphError(f"Frame ID {frame_id} already in use.")

        frame = Frame(graph=self, frame_id=frame_id)
        self._frames[frame_id] = frame
        self._path_cache = None
        return frame

    def remove_frame(self, frame: Frame | FrameID) -> None:
        """Remove a frame and all transforms connected to it."""
        frame = self._as_valid_frame(frame)
        to_remove = [
            t
            for (f1, f2), t in self._transforms.items()
            if frame in (f1, f2)
        ]
        for t in to_remove:
            self.remove_transform(t)
        self._frames.pop(frame.frame_id)
        self._path_cache = None

    def add_transform(self, transform: FrameTransform) -> FrameTransform:
        """Add a transform edge between two frames already in the graph."""
        from_frame, to_frame = transform.from_frame, transform.to_frame

        if from_frame not in self._frames.inv or to_frame not in self._frames.inv:
            raise GraphError(
                "Both transform frames must already exist in the graph: "
                f"{from_frame} -> {to_frame}"
            )
        if (from_frame, to_frame) in self._transforms:
            raise GraphError(
                f"Transform between {from_frame} and {to_frame} already exists"
            )
        if (to_frame, from_frame) in self._transforms:
            raise GraphError(
                f"Inverse transform between {to_frame} and {from_frame} already exists"
            )
        if from_frame == to_frame:
            raise GraphError(f"Cannot add self-loop transform on {from_frame}")

        self._transforms[(from_frame, to_frame)] = transform
        self._path_cache = None
        return transform

    def set_transform(self, transform: FrameTransform) -> FrameTransform:
        """Add or replace a transform between two frames.

        Like :meth:`add_transform`, but silently replaces an existing
        transform between the same pair of frames (in either direction).
        """
        from_frame, to_frame = transform.from_frame, transform.to_frame
        key = (from_frame, to_frame)
        inv_key = (to_frame, from_frame)
        if key in self._transforms:
            self._transforms.pop(key)
            self._path_cache = None
        elif inv_key in self._transforms:
            self._transforms.pop(inv_key)
            self._path_cache = None
        return self.add_transform(transform)

    def remove_transform(self, transform: FrameTransform) -> None:
        """Remove a transform edge from the graph."""
        key = (transform.from_frame, transform.to_frame)
        if key not in self._transforms:
            raise GraphError(f"{transform} not found in graph")
        self._transforms.pop(key)
        self._path_cache = None

    def clear_transforms(self) -> None:
        """Remove all transforms from the graph."""
        self._transforms.clear()
        self._path_cache = None

    def path(
        self,
        from_frame: Frame | FrameID,
        to_frame: Frame | FrameID,
    ) -> list[PathStep]:
        """Return the shortest sequence of transform steps between two frames.

        Raises:
            GraphError: If no path exists between the frames.
        """
        if self._path_cache is None:
            self._compute_paths()

        from_frame = self._as_valid_frame(from_frame)
        to_frame = self._as_valid_frame(to_frame)
        if from_frame == to_frame:
            return []

        frame_index = self._path_cache.frame_index
        edge_directions = self._path_cache.directions
        predecessors = self._path_cache.predecessors

        cur_frame = from_frame
        cur_idx = frame_index[cur_frame]
        target_idx = frame_index[to_frame]

        if predecessors[target_idx, cur_idx] < 0:
            raise GraphError(f"No path found from {from_frame} to {to_frame}")

        path: list[PathStep] = []
        while cur_idx != target_idx:
            next_idx = predecessors[target_idx, cur_idx]
            next_frame = frame_index.inv[next_idx]
            invert = bool(edge_directions[next_idx, cur_idx] == -1)
            frames = (next_frame, cur_frame) if invert else (cur_frame, next_frame)
            transform = self._transforms[frames]
            path.append(PathStep(transform=transform, invert=invert))
            cur_frame, cur_idx = next_frame, next_idx

        return path

    def transform(
        self,
        obj: TFrameTransformable,
        to_frame: Frame | FrameID,
    ) -> TFrameTransformable:
        """Transform an object from its current frame to a target frame."""
        path = self.path(obj.frame, to_frame)
        transformed = obj
        for step in path:
            t = step.transform.inv() if step.invert else step.transform
            transformed = t.apply(transformed)
        return transformed

    def show(
        self,
        view: bool = False,
        filename: str = "graph",
        directory: os.PathLike | None = None,
        **kwargs,
    ):
        from pyspace.render import render_graph

        return render_graph(
            self,
            view=view,
            filename=filename,
            directory=directory,
            **kwargs,
        )

    def _as_valid_frame(self, frame: Frame | FrameID) -> Frame:
        if frame in self._frames.inv:
            return frame
        try:
            return self._frames[frame]
        except KeyError:
            raise GraphError(f"{frame} not found in graph")

    def _compute_paths(self) -> None:
        """Compute and cache all-pairs shortest path info."""
        n = len(self._frames)
        frame_index = bidict(
            {frame: i for i, frame in enumerate(self._frames.values())}
        )

        if n == 0 or len(self._transforms) == 0:
            self._path_cache = PathCache(
                frame_index=frame_index,
                directions=np.zeros((n, n), dtype=np.int8),
                predecessors=np.full((n, n), -9999, dtype=int),
            )
            return

        from_indices, to_indices = zip(
            *(
                (frame_index[t.from_frame], frame_index[t.to_frame])
                for t in self._transforms.values()
            )
        )
        directions = np.zeros((n, n), dtype=np.int8)
        directions[to_indices, from_indices] = 1
        _, predecessors = csgraph.shortest_path(
            directions,
            unweighted=True,
            directed=False,
            method="D",
            return_predecessors=True,
        )
        directions[from_indices, to_indices] = -1

        self._path_cache = PathCache(
            frame_index=frame_index,
            directions=directions,
            predecessors=predecessors,
        )

    def __getitem__(self, frame_id: FrameID) -> Frame:
        return self._frames[frame_id]
