"""Frame-transformable protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, TypeVar

if TYPE_CHECKING:
    from pyspace.frame import Frame, FrameID
    from pyspace.transform import FrameTransform


class FrameTransformable(Protocol):
    """An object that lives in a reference frame and can be transformed."""

    @property
    def frame(self) -> Frame: ...

    def apply_frame_transform(self, transform: FrameTransform) -> Self: ...

    def to(self, frame: Frame | FrameID) -> Self:
        return self.frame.graph.transform(self, frame)


TFrameTransformable = TypeVar("TFrameTransformable", bound=FrameTransformable)
