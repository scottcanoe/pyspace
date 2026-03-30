"""Frame transform type."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import RigidTransform, Rotation

from pyspace.exceptions import FrameMismatchError

if TYPE_CHECKING:
    from pyspace.frame import Frame
    from pyspace.protocols import TFrameTransformable


class FrameTransform:
    """A rigid transformation between two reference frames.

    ``FrameTransform`` objects are the edges in a :class:`FrameGraph`.
    """

    def __init__(
        self,
        rigid_transform: RigidTransform,
        from_frame: Frame,
        to_frame: Frame,
    ) -> None:
        self._rigid_transform = rigid_transform
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
        return self._rigid_transform.translation

    @property
    def rotation(self) -> Rotation:
        return self._rigid_transform.rotation

    def as_translation_and_rotation(self) -> tuple[np.ndarray, Rotation]:
        return self._rigid_transform.translation, self._rigid_transform.rotation

    def as_rigid_transform(self) -> RigidTransform:
        return self._rigid_transform

    @staticmethod
    def from_translation_and_rotation(
        translation: ArrayLike,
        rotation: Rotation,
        from_frame: Frame,
        to_frame: Frame,
    ) -> FrameTransform:
        rigid_transform = RigidTransform.from_components(
            rotation=rotation,
            translation=translation,
        )
        return FrameTransform(
            rigid_transform=rigid_transform,
            from_frame=from_frame,
            to_frame=to_frame,
        )

    @staticmethod
    def from_rigid_transform(
        rigid_transform: RigidTransform,
        from_frame: Frame,
        to_frame: Frame,
    ) -> FrameTransform:
        return FrameTransform(
            rigid_transform=rigid_transform,
            from_frame=from_frame,
            to_frame=to_frame,
        )

    def inv(self) -> FrameTransform:
        return FrameTransform(
            rigid_transform=self._rigid_transform.inv(),
            from_frame=self._to_frame,
            to_frame=self._from_frame,
        )

    def apply(self, obj: TFrameTransformable) -> TFrameTransformable:
        if obj.frame != self._from_frame:
            raise FrameMismatchError(
                f"Cannot apply {self}: object is in {obj.frame}, "
                f"expected {self._from_frame}"
            )
        out = obj.apply_frame_transform(self)
        if out.frame != self._to_frame:
            raise FrameMismatchError(
                f"Transform application produced object in {out.frame}, "
                f"expected {self._to_frame}"
            )
        return out

    def __repr__(self) -> str:
        return f"FrameTransform({self._from_frame.frame_id} -> {self._to_frame.frame_id})"
