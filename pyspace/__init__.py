"""Geometry and reference-frame utilities."""

from pyspace.animation import animate_object_poses, interpolate_poses, interpolate_transforms
from pyspace.transform_graph import (
    Frame,
    FrameMismatchError,
    FrameTransform,
    Graph,
    GraphError,
    Location,
    Orientation,
    Pose,
)

# Backward-compatible aliases for older external API names.
FrameGraph = Graph
FrameGraphError = GraphError
FrameNotFoundError = GraphError
TransformNotFoundError = GraphError
Transform = FrameTransform

__all__ = [
    "animate_object_poses",
    "Frame",
    "FrameGraph",
    "FrameGraphError",
    "FrameMismatchError",
    "FrameNotFoundError",
    "FrameTransform",
    "Location",
    "Orientation",
    "Pose",
    "Transform",
    "TransformNotFoundError",
    "interpolate_poses",
    "interpolate_transforms",
]

