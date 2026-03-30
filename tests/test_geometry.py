"""Tests for geometric types: Location, Displacement, Orientation, Pose."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import FrameGraph, FrameTransform, Location, Orientation, Pose
from pyspace.geometry import Displacement


class TestLocation(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")

    def test_create_via_frame(self):
        loc = self.a.location([1, 2, 3])
        self.assertIs(loc.frame, self.a)
        self.assertTrue(np.allclose(loc.as_array(), [1, 2, 3]))

    def test_create_directly(self):
        loc = Location([4, 5, 6], self.a)
        self.assertTrue(np.allclose(loc.as_array(), [4, 5, 6]))

    def test_from_array(self):
        loc = Location.from_array([7, 8, 9], self.a)
        self.assertTrue(np.allclose(loc.as_array(), [7, 8, 9]))

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            Location([1, 2], self.a)

    def test_as_array_returns_copy(self):
        loc = self.a.location([1, 2, 3])
        arr = loc.as_array()
        arr[0] = 999
        self.assertEqual(loc.as_array()[0], 1.0)

    def test_to_shorthand(self):
        b = self.graph.add_frame("b")
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.array([10.0, 0.0, 0.0]), Rotation.identity(), self.a, b,
        ))
        result = self.a.location([1, 2, 3]).to(b)
        self.assertIs(result.frame, b)
        self.assertTrue(np.allclose(result.as_array(), [11, 2, 3]))

    def test_repr(self):
        self.assertIn("Location", repr(self.a.location([1, 2, 3])))


class TestDisplacement(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")

    def test_create_via_frame(self):
        d = self.a.displacement([1, 0, 0])
        self.assertIs(d.frame, self.a)
        self.assertTrue(np.allclose(d.as_array(), [1, 0, 0]))

    def test_ignores_translation(self):
        """Only rotation is applied; the large translation has no effect."""
        b = self.graph.add_frame("b")
        rot = Rotation.from_euler("z", 90, degrees=True)
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.array([100, 200, 300]), rot, self.a, b,
        ))
        result = self.a.displacement([1.0, 0.0, 0.0]).to(b)
        self.assertTrue(np.allclose(result.as_array(), [0, 1, 0], atol=1e-10))

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            Displacement([1, 2, 3, 4], self.a)

    def test_as_array_returns_copy(self):
        d = self.a.displacement([1, 0, 0])
        arr = d.as_array()
        arr[0] = 999
        self.assertEqual(d.as_array()[0], 1.0)


class TestOrientation(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")

    def test_create_via_frame(self):
        rot = Rotation.from_euler("z", 90, degrees=True)
        ori = self.a.orientation(rot)
        self.assertIs(ori.frame, self.a)
        self.assertTrue(np.allclose(ori.as_matrix(), rot.as_matrix()))

    def test_from_euler(self):
        ori = Orientation.from_euler(self.a, "z", 90, degrees=True)
        expected = Rotation.from_euler("z", 90, degrees=True)
        self.assertTrue(np.allclose(ori.as_matrix(), expected.as_matrix()))

    def test_from_matrix_roundtrip(self):
        mat = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        ori = Orientation.from_matrix(self.a, mat)
        self.assertTrue(np.allclose(ori.as_matrix(), mat))

    def test_from_quat_roundtrip(self):
        quat = np.array([1, 0, 0, 0], dtype=float)
        ori = Orientation.from_quat(self.a, quat, scalar_first=True)
        self.assertTrue(np.allclose(ori.as_quat(), quat))

    def test_from_rotation(self):
        rot = Rotation.from_euler("x", 45, degrees=True)
        ori = Orientation.from_rotation(self.a, rot)
        self.assertTrue(np.allclose(ori.as_rotation().as_matrix(), rot.as_matrix()))

    def test_as_euler(self):
        ori = Orientation.from_euler(self.a, "z", 90, degrees=True)
        self.assertTrue(np.allclose(ori.as_euler("xyz", degrees=True), [0, 0, 90]))

    def test_transform_composes_rotation(self):
        b = self.graph.add_frame("b")
        rot_t = Rotation.from_euler("z", 90, degrees=True)
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.zeros(3), rot_t, self.a, b,
        ))
        rot_obj = Rotation.from_euler("z", 45, degrees=True)
        ori_b = self.a.orientation(rot_obj).to(b)
        expected = (rot_t * rot_obj).as_matrix()
        self.assertTrue(np.allclose(ori_b.as_matrix(), expected))


class TestPose(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")

    def test_create_via_frame(self):
        pose = self.a.pose([1, 2, 3], Rotation.identity())
        self.assertIs(pose.frame, self.a)
        self.assertTrue(np.allclose(pose.location.as_array(), [1, 2, 3]))
        self.assertTrue(np.allclose(pose.orientation.as_matrix(), np.eye(3)))

    def test_mismatched_frames_raises(self):
        b = self.graph.add_frame("b")
        with self.assertRaises(ValueError):
            Pose(self.a.location([0, 0, 0]), b.orientation(Rotation.identity()))

    def test_transform(self):
        b = self.graph.add_frame("b")
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.array([5.0, 0.0, 0.0]), Rotation.identity(), self.a, b,
        ))
        pose_b = self.a.pose([1, 0, 0], Rotation.identity()).to(b)
        self.assertIs(pose_b.frame, b)
        self.assertTrue(np.allclose(pose_b.location.as_array(), [6, 0, 0]))

    def test_round_trip(self):
        b = self.graph.add_frame("b")
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.array([1.0, 2.0, 3.0]),
            Rotation.from_euler("xyz", [30, 45, 60], degrees=True),
            self.a, b,
        ))
        original = self.a.pose([1, 2, 3], Rotation.from_euler("z", 90, degrees=True))
        roundtrip = original.to(b).to(self.a)
        self.assertTrue(np.allclose(
            roundtrip.location.as_array(), original.location.as_array(),
        ))
        self.assertTrue(np.allclose(
            roundtrip.orientation.as_matrix(), original.orientation.as_matrix(),
        ))

    def test_repr(self):
        self.assertIn("Pose", repr(self.a.pose([0, 0, 0], Rotation.identity())))
