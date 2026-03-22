import unittest

import numpy as np
import pytest

from pyspace.tree import (
    Edge,
    Graph,
    NameInUseError,
    Node,
    NodeID,
    NodeName,
    NodesIncompatibleError,
    NoPathError,
    TreeInvariantError,
)


class TestEmptyTree(unittest.TestCase):
    
    def test_empty_tree_invariants(self):
        tree = Graph()
        self.assertIsNone(tree.root)
        self.assertEqual(tree.size, 0)
        self.assertEqual(tree.nodes, ())
        self.assertEqual(tree.edges, ())


class TestCreateRoot(unittest.TestCase):

    def test_creates_node_sets_root_when_tree_has_no_root(self):
        tree = Graph()
        tree.create_node(parent=None)
        self.assertIsNotNone(tree.root)

    def test_create_node_raises_error_when_parent_arg_is_none_and_tree_has_root(self):
        tree = Graph()
        tree.create_node(parent=None)
        with self.assertRaises(TreeInvariantError):
            tree.create_node(parent=None)


class TestOneNodeTree(unittest.TestCase):
    
    def test_one_node_tree_invariants(self):
        tree = Graph()
        tree.create_node(parent=None)
        self.assertIsNone(tree.root.parent)
        self.assertEqual(tree.size, 1)
        self.assertEqual(tree.nodes, (tree.root,))
        self.assertEqual(tree.edges, ())



class TestCreateNode(unittest.TestCase):

    def test_tree_growsadds_node(self):
        tree = Graph()
        num_calls = 5
        expected_sizes = np.arange(1, 1 + num_calls)
        actual_sizes = []
        parent = None
        for _ in range(num_calls):
            parent = tree.create_node(parent=parent)
            actual_sizes.append(tree.size)
            self.assertEqual(actual_sizes, expected_sizes)


class TestCreateNode(unittest.TestCase):
    def test_create_first_child(self):
        tree = Graph()
        parent = tree.create_node(parent=None, name=NodeName("parent"))
        child = tree.create_node(parent=tree.root, name=NodeName("child"))
        self.assertEqual(tree.size, 2)
        self.assertEqual(tree.nodes, (parent, child))
        self.assertEqual(tree.edges, (Edge(parent=parent, child=child),))









# def test_tree_parent_children_and_matrix():
#     t = NodeTree()
#     root = t.create_root()
#     a = t.add_child(root)
#     b = t.add_child(root)
#     c = t.add_child(a)

#     assert root.parent is None
#     assert a.parent is root
#     assert b.parent is root
#     assert c.parent is a

#     assert set(root.children) == {a, b}
#     assert a.children == (c,)
#     assert b.children == ()
#     assert c.children == ()

#     assert t.i == (0, 0, 1)
#     assert t.j == (1, 2, 3)
#     assert t.connections.shape == (4, 4)
#     assert t.connections.nnz == 3

#     assert root.depth() == 0
#     assert a.depth() == 1
#     assert c.depth() == 2

#     # Shortest path from root along directed edges
#     assert int(t.dist[0, 3]) == 2
#     assert int(t.predecessors[0, 3]) == 1

