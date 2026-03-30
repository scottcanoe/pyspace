"""Geometry and reference-frame utilities."""

from pyspace.exceptions import FrameMismatchError, GraphError
from pyspace.frame import Frame, FrameID
from pyspace.geometry import Displacement, Location, Orientation, Pose
from pyspace.graph import FrameGraph, PathStep
from pyspace.protocols import FrameTransformable, TFrameTransformable
from pyspace.transform import FrameTransform

# Backward-compatible aliases for older external API names.
FrameGraphError = GraphError
FrameNotFoundError = GraphError
TransformNotFoundError = GraphError
Transform = FrameTransform

__all__ = [
    "Displacement",
    "Frame",
    "FrameGraph",
    "FrameGraphError",
    "FrameID",
    "FrameMismatchError",
    "FrameNotFoundError",
    "FrameTransform",
    "FrameTransformable",
    "GraphError",
    "Location",
    "Orientation",
    "PathStep",
    "Pose",
    "TFrameTransformable",
    "Transform",
    "TransformNotFoundError",
]
