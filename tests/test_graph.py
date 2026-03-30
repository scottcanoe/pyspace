"""Tests for FrameGraph: frame/transform management, path finding, and object transforms."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import FrameGraph, FrameTransform, GraphError


class TestAddFrame(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()

    def test_with_id(self):
        frame = self.graph.add_frame("world")
        self.assertEqual(frame.frame_id, "world")
        self.assertIs(frame.graph, self.graph)

    def test_without_id(self):
        frame = self.graph.add_frame()
        self.assertIsNotNone(frame.frame_id)
        self.assertGreater(len(frame.frame_id), 0)

    def test_duplicate_id_raises(self):
        self.graph.add_frame("world")
        with self.assertRaises(GraphError):
            self.graph.add_frame("world")

    def test_frames_property_returns_snapshot(self):
        a = self.graph.add_frame("a")
        frames = self.graph.frames
        self.assertIn("a", frames)
        self.assertIs(frames["a"], a)


class TestRemoveFrame(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")

    def test_by_object(self):
        self.graph.remove_frame(self.a)
        self.assertNotIn("a", self.graph.frames)

    def test_by_id(self):
        self.graph.remove_frame("a")
        self.assertNotIn("a", self.graph.frames)

    def test_cascades_to_transforms(self):
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.zeros(3), Rotation.identity(), self.a, self.b,
        ))
        self.graph.remove_frame(self.a)
        self.assertEqual(len(self.graph.transforms), 0)

    def test_nonexistent_raises(self):
        with self.assertRaises(GraphError):
            self.graph.remove_frame("ghost")


class TestAddTransform(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")

    def _link(self, translation=None, from_frame=None, to_frame=None):
        return FrameTransform.from_translation_and_rotation(
            translation if translation is not None else np.zeros(3),
            Rotation.identity(),
            from_frame or self.a,
            to_frame or self.b,
        )

    def test_basic(self):
        self.graph.add_transform(self._link(np.array([1.0, 0, 0])))
        self.assertEqual(len(self.graph.transforms), 1)

    def test_duplicate_raises(self):
        self.graph.add_transform(self._link())
        with self.assertRaises(GraphError):
            self.graph.add_transform(self._link(np.ones(3)))

    def test_reverse_duplicate_raises(self):
        self.graph.add_transform(self._link())
        with self.assertRaises(GraphError):
            self.graph.add_transform(self._link(from_frame=self.b, to_frame=self.a))

    def test_self_loop_raises(self):
        with self.assertRaises(GraphError):
            self.graph.add_transform(self._link(from_frame=self.a, to_frame=self.a))

    def test_foreign_frame_raises(self):
        other = FrameGraph()
        foreign = other.add_frame("x")
        with self.assertRaises(GraphError):
            self.graph.add_transform(self._link(to_frame=foreign))


class TestRemoveTransform(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")
        self.t = FrameTransform.from_translation_and_rotation(
            np.zeros(3), Rotation.identity(), self.a, self.b,
        )

    def test_remove(self):
        self.graph.add_transform(self.t)
        self.graph.remove_transform(self.t)
        self.assertEqual(len(self.graph.transforms), 0)

    def test_nonexistent_raises(self):
        with self.assertRaises(GraphError):
            self.graph.remove_transform(self.t)


class TestSetTransform(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")

    def test_adds_when_none_exists(self):
        t = FrameTransform.from_translation_and_rotation(
            np.array([1.0, 0, 0]), Rotation.identity(), self.a, self.b,
        )
        self.graph.set_transform(t)
        self.assertEqual(len(self.graph.transforms), 1)

    def test_replaces_same_direction(self):
        t1 = FrameTransform.from_translation_and_rotation(
            np.array([1.0, 0, 0]), Rotation.identity(), self.a, self.b,
        )
        t2 = FrameTransform.from_translation_and_rotation(
            np.array([2.0, 0, 0]), Rotation.identity(), self.a, self.b,
        )
        self.graph.set_transform(t1)
        self.graph.set_transform(t2)
        self.assertEqual(len(self.graph.transforms), 1)
        loc = self.a.location([0, 0, 0]).to(self.b)
        self.assertTrue(np.allclose(loc.as_array(), [2, 0, 0]))

    def test_replaces_reverse_direction(self):
        t1 = FrameTransform.from_translation_and_rotation(
            np.array([1.0, 0, 0]), Rotation.identity(), self.a, self.b,
        )
        t2 = FrameTransform.from_translation_and_rotation(
            np.array([5.0, 0, 0]), Rotation.identity(), self.b, self.a,
        )
        self.graph.set_transform(t1)
        self.graph.set_transform(t2)
        self.assertEqual(len(self.graph.transforms), 1)
        loc = self.b.location([0, 0, 0]).to(self.a)
        self.assertTrue(np.allclose(loc.as_array(), [5, 0, 0]))


class TestClearTransforms(unittest.TestCase):
    def test_clear(self):
        graph = FrameGraph()
        a, b = graph.add_frame("a"), graph.add_frame("b")
        graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.zeros(3), Rotation.identity(), a, b,
        ))
        graph.clear_transforms()
        self.assertEqual(len(graph.transforms), 0)
        self.assertEqual(len(graph.frames), 2)


class TestPath(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")
        self.t = FrameTransform.from_translation_and_rotation(
            np.array([1.0, 2.0, 3.0]), Rotation.identity(), self.a, self.b,
        )
        self.graph.add_transform(self.t)

    def test_same_frame_returns_empty(self):
        self.assertEqual(self.graph.path(self.a, self.a), [])

    def test_direct_forward(self):
        path = self.graph.path(self.a, self.b)
        self.assertEqual(len(path), 1)
        self.assertIs(path[0].transform, self.t)
        self.assertFalse(path[0].invert)

    def test_direct_inverse(self):
        path = self.graph.path(self.b, self.a)
        self.assertEqual(len(path), 1)
        self.assertIs(path[0].transform, self.t)
        self.assertTrue(path[0].invert)

    def test_multi_hop(self):
        c = self.graph.add_frame("c")
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.zeros(3), Rotation.identity(), self.b, c,
        ))
        path = self.graph.path(self.a, c)
        self.assertEqual(len(path), 2)

    def test_disconnected_raises(self):
        self.graph.add_frame("c")
        with self.assertRaises(GraphError):
            self.graph.path(self.a, "c")

    def test_by_frame_id(self):
        self.assertEqual(len(self.graph.path("a", "b")), 1)


class TestTransformObject(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()
        self.a = self.graph.add_frame("a")
        self.b = self.graph.add_frame("b")

    def test_translate_location(self):
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.array([1.0, 2.0, 3.0]), Rotation.identity(), self.a, self.b,
        ))
        result = self.graph.transform(self.a.location([0, 0, 0]), self.b)
        self.assertIs(result.frame, self.b)
        self.assertTrue(np.allclose(result.as_array(), [1.0, 2.0, 3.0]))

    def test_chain_translation(self):
        c = self.graph.add_frame("c")
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.array([1.0, 0.0, 0.0]), Rotation.identity(), self.a, self.b,
        ))
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.array([0.0, 1.0, 0.0]), Rotation.identity(), self.b, c,
        ))
        result = self.graph.transform(self.a.location([0, 0, 0]), c)
        self.assertTrue(np.allclose(result.as_array(), [1.0, 1.0, 0.0]))

    def test_round_trip_with_rotation(self):
        self.graph.add_transform(FrameTransform.from_translation_and_rotation(
            np.array([1.0, 2.0, 3.0]),
            Rotation.from_euler("z", 45, degrees=True),
            self.a, self.b,
        ))
        loc = self.a.location([1, 2, 3])
        roundtrip = self.graph.transform(self.graph.transform(loc, self.b), self.a)
        self.assertTrue(np.allclose(roundtrip.as_array(), loc.as_array()))
        self.assertIs(roundtrip.frame, self.a)


class TestGetItem(unittest.TestCase):
    def setUp(self):
        self.graph = FrameGraph()

    def test_lookup(self):
        a = self.graph.add_frame("a")
        self.assertIs(self.graph["a"], a)

    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            self.graph["nope"]
