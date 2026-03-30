"""Tests for FrameTransform: creation, inversion, and application."""

import unittest

import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation

from pyspace import FrameGraph, FrameMismatchError, FrameTransform


class TestCreation(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")

    def test_from_translation_and_rotation(self):
        t = FrameTransform.from_translation_and_rotation(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation=Rotation.identity(),
            from_frame=self.a,
            to_frame=self.b,
        )
        self.assertIs(t.from_frame, self.a)
        self.assertIs(t.to_frame, self.b)
        self.assertTrue(np.allclose(t.translation, [1, 2, 3]))

    def test_from_rigid_transform(self):
        rt = RigidTransform.from_components(
            rotation=Rotation.identity(),
            translation=np.zeros(3),
        )
        t = FrameTransform.from_rigid_transform(rt, self.a, self.b)
        self.assertIs(t.from_frame, self.a)
        self.assertIs(t.to_frame, self.b)

    def test_as_translation_and_rotation(self):
        rot = Rotation.from_euler("z", 45, degrees=True)
        t = FrameTransform.from_translation_and_rotation(
            np.array([1.0, 2.0, 3.0]), rot, self.a, self.b,
        )
        trans, r = t.as_translation_and_rotation()
        self.assertTrue(np.allclose(trans, [1, 2, 3]))
        self.assertTrue(np.allclose(r.as_matrix(), rot.as_matrix()))

    def test_as_rigid_transform(self):
        t = FrameTransform.from_translation_and_rotation(
            np.array([1.0, 0, 0]), Rotation.identity(), self.a, self.b,
        )
        self.assertIsInstance(t.as_rigid_transform(), RigidTransform)


class TestInversion(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")

    def test_swaps_frames(self):
        t = FrameTransform.from_translation_and_rotation(
            np.array([1, 0, 0]), Rotation.identity(), self.a, self.b,
        )
        inv = t.inv()
        self.assertIs(inv.from_frame, self.b)
        self.assertIs(inv.to_frame, self.a)

    def test_negates_pure_translation(self):
        t = FrameTransform.from_translation_and_rotation(
            np.array([1.0, 2.0, 3.0]), Rotation.identity(), self.a, self.b,
        )
        self.assertTrue(np.allclose(t.inv().translation, [-1, -2, -3]))

    def test_double_inv_is_identity(self):
        t = FrameTransform.from_translation_and_rotation(
            np.array([1, 2, 3]),
            Rotation.from_euler("xyz", [10, 20, 30], degrees=True),
            self.a, self.b,
        )
        roundtrip = t.inv().inv()
        self.assertTrue(np.allclose(roundtrip.translation, t.translation))
        self.assertTrue(np.allclose(
            roundtrip.rotation.as_matrix(), t.rotation.as_matrix(),
        ))


class TestApply(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")

    def test_translates_location(self):
        t = FrameTransform.from_translation_and_rotation(
            np.array([10.0, 0.0, 0.0]), Rotation.identity(), self.a, self.b,
        )
        result = t.apply(self.a.location([1, 2, 3]))
        self.assertIs(result.frame, self.b)
        self.assertTrue(np.allclose(result.as_array(), [11, 2, 3]))

    def test_rotates_location(self):
        rot = Rotation.from_euler("z", 90, degrees=True)
        t = FrameTransform.from_translation_and_rotation(
            np.zeros(3), rot, self.a, self.b,
        )
        result = t.apply(self.a.location([1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(result.as_array(), [0, 1, 0], atol=1e-10))

    def test_wrong_frame_raises(self):
        c = self.graph.add_frame("c")
        t = FrameTransform.from_translation_and_rotation(
            np.zeros(3), Rotation.identity(), self.a, self.b,
        )
        with self.assertRaises(FrameMismatchError):
            t.apply(c.location([0, 0, 0]))

    def test_repr(self):
        t = FrameTransform.from_translation_and_rotation(
            np.zeros(3), Rotation.identity(), self.a, self.b,
        )
        self.assertIn("a", repr(t))
        self.assertIn("b", repr(t))
