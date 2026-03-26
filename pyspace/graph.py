"""Tree of nodes backed by a sparse adjacency matrix (pytransform3d-style).

Relationships are stored as directed parent→child edges in parallel index lists
``i`` (parent column index) and ``j`` (child column index), then folded into a
CSR matrix ``connections`` for graph algorithms — mirroring
:class:`pytransform3d.transform_manager.TransformGraphBase`.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    NewType,
    Protocol,
    Self,
)

import numpy as np
from bidict import bidict
from numpy._typing import ArrayLike
from scipy.sparse import csgraph
from scipy.spatial.transform import RigidTransform as RigidTransform
from scipy.spatial.transform import Rotation as Rotation

FrameID = NewType("FrameID", str)
TransformID = NewType("TransformID", str)


class GraphError(Exception):
    pass




class Frame:
    """A reference frame."""

    def __init__(
        self,
        graph: Graph,
        frame_id: FrameID,
    ) -> None:
        self._graph = graph
        self._frame_id = frame_id

    @property
    def graph(self) -> Graph:
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


class FrameTransform:
    """
    Transforms from pose.frame to...
    """
    def __init__(
        self,
        transform: RigidTransform,
        from_frame: Frame,
        to_frame: Frame,
    ) -> None:
        self._transform = transform
        self._from_frame = from_frame
        self._to_frame = to_frame

    @property
    def from_frame(self) -> Frame:
        return self._from_frame
    
    @property
    def to_frame(self) -> Frame:
        return self._to_frame

    @property
    def translation(self) -> np.ndarray:
        return self._transform.translation

    @property
    def rotation(self) -> Rotation:
        return self._transform.rotation
    
    def as_components(self) -> tuple[np.ndarray, Rotation]:
        return self._transform.translation, self._transform.rotation
    
    def as_rigid_transform(self) -> RigidTransform:
        return self._transform
    
    @staticmethod
    def from_components(
        translation: ArrayLike,
        rotation: Rotation,
        from_frame: Frame,
        to_frame: Frame,
    ) -> FrameTransform:
        transform = RigidTransform.from_components(
            rotation=rotation, translation=translation,
        )
        return FrameTransform(
            transform=transform,
            from_frame=from_frame,
            to_frame=to_frame,
        )
    
    @staticmethod
    def from_rigid_transform(
        transform: RigidTransform,
        from_frame: Frame,
        to_frame: Frame,
        ) -> FrameTransform:
        return FrameTransform(
            transform=transform,
            from_frame=from_frame,
            to_frame=to_frame,
        )

    def inv(self) -> FrameTransform:
        return FrameTransform(
            transform=self._transform.inv(),
            from_frame=self._to_frame,
            to_frame=self._from_frame,
        )

    def apply(self, obj: FrameTransformable) -> FrameTransformable:
        assert obj.frame == self._from_frame
        out = obj.apply_frame_transform(self)
        assert out.frame == self._to_frame
        return out
        
    def __repr__(self) -> str:
        return f"FrameTransform({self._from_frame.frame_id} -> {self._to_frame.frame_id})"



@dataclass(frozen=True)
class PathStep:
    transform: FrameTransform
    inverse: bool


@dataclass
class GraphCache:
    
    frame_index: bidict[Frame, int]
    adjacency_matrix: np.ndarray
    predecessors: np.ndarray


class Graph:
    def __init__(self) -> None:
        # reference frames (nodes)
        self._frame_id_generator: Callable[[], FrameID] = lambda: FrameID(uuid.uuid4())
        self._frames: set[Frame] = set()
        self._frame_id_to_frame: dict[FrameID, Frame] = {}
        
        # refernce frame transforms (edges)
        self._transforms: set[FrameTransform] = set()
        self._frames_to_transform: dict[tuple[Frame, Frame], FrameTransform] = {}

        # shortest paths
        self._cache: GraphCache | None = None

    @property
    def frames(self) -> set[Frame]:
        return set(self._frames)

    @property
    def transforms(self) -> set[FrameTransform]:
        return set(self._transforms)

    @property
    def size(self) -> int:
        return len(self._frames)

    def add_frame(
        self,
        frame_id: FrameID | None = None,
    ) -> Frame:
        """Add a frame to the graph."""

        # Check `uid` won't clash.
        frame_id = self._frame_id_generator() if frame_id is None else frame_id
        if frame_id in self._frame_id_to_frame:
            raise GraphError(f"Frame ID {frame_id} already in use.")
        
        frame = Frame(graph=self, frame_id=frame_id)
        self._frames.add(frame)
        self._frame_id_to_frame[frame_id] = frame
        self._cache = None
        return frame

    def remove_frame(self, frame: Frame | FrameID) -> None:
        frame = self._as_valid_frame(frame)
        self._frames.discard(frame)
        self._frame_id_to_frame.pop(frame.frame_id)
        self._cache = None
        transforms_to_remove = []
        for (from_frame, to_frame), tform in self._frames_to_transform.items():
            if frame in (from_frame, to_frame):
                transforms_to_remove.append(tform)
        for transform in transforms_to_remove:
            self.remove_transform(transform)

    def add_transform(self, tform: FrameTransform) -> FrameTransform:

        from_frame = tform.from_frame
        to_frame = tform.to_frame
        assert from_frame in self._frames and to_frame in self._frames
        
        if (from_frame, to_frame) in self._frames_to_transform:
            raise GraphError(f"Transform between {from_frame} and {to_frame} already exists")
        
        if (to_frame, from_frame) in self._frames_to_transform:
            raise GraphError(f"Inverse transform between {to_frame} and {from_frame} already exists")
        
        self._transforms.add(tform)
        self._frames_to_transform[(from_frame, to_frame)] = tform
        self._cache = None
        return tform

    def remove_transform(self, tform: FrameTransform) -> None:
        
        self._transforms.remove(tform)
        self._frames_to_transform.pop((tform.from_frame, tform.to_frame))
        # Remove inverse transform if it exists.
        self._frames_to_transform.pop((tform.to_frame, tform.from_frame), None)
        self._cache = None
    
    def clear_transforms(self) -> None:
        self._transforms.clear()
        self._frames_to_transform.clear()
        self._cache = None

    def shortest_path(
        self,
        from_frame: Frame | FrameID,
        to_frame: Frame | FrameID,
    ) -> list[PathStep]:
        """Return transforms and inverse flags for path from start to end frames.

        Returned transforms don't include the start transform.

        Raises:
            RuntimeError: If start and end aren't connected.

        """
        if not self._cache:
            self._compute_shortest_paths()
        
        from_frame = self._as_valid_frame(from_frame)
        to_frame = self._as_valid_frame(to_frame)
        if from_frame == to_frame:
            return []

        frame_index: bidict[Frame, int] = self._cache.frame_index
        adjacency_matrix: np.ndarray = self._cache.adjacency_matrix
        predecessors: np.ndarray = self._cache.predecessors

        cur_frame: Frame = from_frame
        cur_frame_index = frame_index[cur_frame]
        target_frame_index = frame_index[to_frame]
                
        if predecessors[target_frame_index, cur_frame_index] < 0:
            raise GraphError(f"No path found from {from_frame} to {to_frame}")

        path: list[PathStep] = []
        while cur_frame_index != target_frame_index:
            next_frame_index = predecessors[target_frame_index, cur_frame_index]
            next_frame = frame_index.inv[next_frame_index]
            inverse = bool(adjacency_matrix[next_frame_index, cur_frame_index] == -1)
            frames = (next_frame, cur_frame) if inverse else (cur_frame, next_frame)
            tform = self._frames_to_transform[frames]
            step = PathStep(transform=tform, inverse=inverse)
            path.append(step)
            cur_frame, cur_frame_index = next_frame, next_frame_index

        return path

    def transform(
        self,
        obj: FrameTransformable,
        to_frame: Frame | FrameID,
    ) -> FrameTransformable:
        """Transform an object from its current frame to a target frame.
        
        TODO: Cache transforms. At least for the source and target frames, possibly
        for the intermediate transforms as well.
        """
        path: list[PathStep] = self.shortest_path(obj.frame, to_frame)
        transformed = obj
        for step in path:
            tform = step.transform
            if step.inverse:
                tform = tform.inv()
            transformed = tform.apply(transformed)
        return transformed


    def show(
        self,
        view: bool = False,
        filename: str = "graph",
        directory: os.PathLike | None = None,
        **kwargs,
    ) -> None:
        from pyspace.render import render_graph
        return render_graph(
            self,
            view=view,
            filename=filename,
            directory=directory,
            **kwargs,
        )

    def _as_valid_frame(self, frame: Frame | FrameID) -> Frame:
        if frame in self._frames:
            return frame
        try:
            return self._frame_id_to_frame[frame]
        except KeyError:
            raise GraphError(f"Frame {frame} not found in graph")
    

    def _compute_shortest_paths(self) -> None:
        """Compute and store all shortest path info."""
        
        if len(self._frames) == 0:
            self._cache = GraphCache(
                frame_index=bidict(),
                adjacency_matrix=np.zeros((0, 0), dtype=int),
                predecessors=np.empty((0, 0), dtype=int),
            )
            return

        frame_index = bidict({frame: i for i, frame in enumerate(self._frames)})
        adjacency_matrix = np.zeros((len(self._frames), len(self._frames)), dtype=int)
        for tform in self._transforms:
            from_index = frame_index[tform.from_frame]
            to_index = frame_index[tform.to_frame]
            adjacency_matrix[to_index, from_index] = 1
            adjacency_matrix[from_index, to_index] = -1
        
        _, predecessors = csgraph.shortest_path(
            np.clip(adjacency_matrix, 0, 1),
            unweighted=True,
            directed=False,
            method="D",
            return_predecessors=True,
        )
        
        self._cache = GraphCache(
            frame_index=frame_index,
            adjacency_matrix=adjacency_matrix,
            predecessors=predecessors,
        )
                
    def __getitem__(self, frame_id: FrameID) -> Frame:
        return self._frame_id_to_frame[frame_id]


"""
--------------------------------------------------------------------------------------
FrameTransformable Protocol (entities)
--------------------------------------------------------------------------------------
"""


class FrameTransformable(Protocol):
    @property
    def frame(self) -> Frame:
        pass

    def apply_frame_transform(self, transform: FrameTransform) -> Self:
        pass

    def to(self, frame: Frame | FrameID) -> Self:
        return self.frame.graph.transform(self, frame)


class Location(FrameTransformable):
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

    def __array__(self, dtype: np.dtype | None = None, copy: bool = True) -> np.ndarray:
        return self._array.copy() if copy else self._array

    def __repr__(self) -> str:
        return f"Location({self._array}, frame={self._frame.frame_id})"


class Displacement(FrameTransformable):
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

    def __array__(self, dtype: np.dtype | None = None, copy: bool = True) -> np.ndarray:
        return self._array.copy() if copy else self._array

    def __repr__(self) -> str:
        return f"Displacement({self._array}, frame={self._frame.frame_id})"


class Orientation(FrameTransformable):
    def __init__(self, rotation: Rotation, frame: Frame) -> None:
        self._rotation = rotation
        self._frame = frame
    
    @property
    def frame(self) -> Frame:
        return self._frame

    def as_euler(
        self,
        dims: str,
        degrees: bool = False,
    ) -> np.ndarray:
        return self._rotation.as_euler(dims, degrees)
    
    def as_matrix(self) -> np.ndarray:
        return self._rotation.as_matrix()

    def as_rotation(self) -> Rotation:
        return self._rotation

    def as_quat(self, scalar_first: bool = True) -> np.ndarray:
        return self._rotation.as_quat(scalar_first=scalar_first)

    @staticmethod
    def from_euler(
        dims: str,
        angles: ArrayLike,
        frame: Frame,
        degrees: bool = False,
    ) -> Orientation:
        return Orientation(Rotation.from_euler(dims, angles, degrees), frame)

    @staticmethod
    def from_matrix(matrix: ArrayLike, frame: Frame) -> Orientation:
        return Orientation(Rotation.from_matrix(matrix), frame)

    @staticmethod
    def from_rotation(rotation: Rotation, frame: Frame) -> Orientation:
        return Orientation(rotation, frame)

    @staticmethod
    def from_quat(
        quat: ArrayLike,
        frame: Frame,
        scalar_first: bool = True,
    ) -> Orientation:
        return Orientation(
            Rotation.from_quat(quat, scalar_first=scalar_first), frame,
        )

    def apply_frame_transform(self, transform: FrameTransform) -> Orientation:
        return Orientation(
            transform.rotation * self._rotation,
            transform.to_frame,
        )

    def __repr__(self) -> str:
        return f"Orientation({self._rotation}, frame={self._frame.frame_id})"



class Pose(FrameTransformable):
    def __init__(self, location: Location, orientation: Orientation) -> None:
        assert location.frame == orientation.frame
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






if __name__ == "__main__":

    # Create a graph with reference frames A, B, and C.
    graph = Graph()
    A = graph.add_frame("A")
    B = graph.add_frame("B")
    C = graph.add_frame("C")

    # Define the location and orientation of frame B relative to frame A,
    # and add the corresponding transform to the graph.
    B_loc = np.array([0, 0, 0])
    B_ori = Rotation.from_euler("xyz", [90, 0, 0], degrees=True)
    B_to_A = FrameTransform.from_components(
        translation=B_loc,
        rotation=B_ori,
        from_frame=B,
        to_frame=A,
    )
    graph.add_transform(B_to_A)

    # Define the location and orientation of frame C relative to frame B,
    # and add the corresponding transform to the graph.
    C_loc = np.array([0, 0, 0])
    C_ori = Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    C_to_B = FrameTransform.from_components(
        translation=C_loc,
        rotation=C_ori,
        from_frame=C,
        to_frame=B,
    )
    graph.add_transform(C_to_B)
    
    p_rel_C = Location([0, 0, -1], frame=C)
    p_rel_B = p_rel_C.to(B)
    p_rel_A = p_rel_B.to(A)

    p_rel_C_round_trip = p_rel_A.to(C)

    print(f"p[C]: {p_rel_C}")
    print(f"p[B]: {p_rel_B}")
    print(f"p[A]: {p_rel_A}")
    print(f"p[C] (round-trip): {p_rel_C_round_trip}")

    assert np.allclose(p_rel_C_round_trip.as_array(), p_rel_C.as_array())
    assert p_rel_C_round_trip.frame == C

