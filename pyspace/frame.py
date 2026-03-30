"""Reference frame type."""

from __future__ import annotations

from typing import TYPE_CHECKING, NewType

from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from pyspace.geometry import Displacement, Location, Orientation, Pose

if TYPE_CHECKING:
    from pyspace.graph import FrameGraph

FrameID = NewType("FrameID", str)


class Frame:
    """A reference frame.

    ``Frame`` objects are the nodes in a :class:`FrameGraph`.

    Do not create them directly — use :meth:`FrameGraph.add_frame`.
    """

    def __init__(self, graph: FrameGraph, frame_id: FrameID) -> None:
        self._graph = graph
        self._frame_id = frame_id

    @property
    def graph(self) -> FrameGraph:
        return self._graph

    @property
    def frame_id(self) -> FrameID:
        return self._frame_id

    def location(self, array: ArrayLike) -> Location:
        return Location(array, self)

    def displacement(self, array: ArrayLike) -> Displacement:
        return Displacement(array, self)

    def orientation(self, rotation: Rotation) -> Orientation:
        return Orientation(rotation, self)

    def pose(self, translation: ArrayLike, rotation: Rotation) -> Pose:
        return Pose(
            location=Location(translation, self),
            orientation=Orientation(rotation, self),
        )

    def __repr__(self) -> str:
        return f"Frame({self.frame_id})"
