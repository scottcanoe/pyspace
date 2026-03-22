import unittest
from unittest.mock import MagicMock

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


class TestNodeRelationalData(unittest.TestCase):
    
    def test_relational_data_correct(self):
        tree = MagicMock()
        node_id = NodeID(0)
        node_name = NodeName("root")
        node = Node(graph=tree, id=node_id, name=node_name)
        self.assertIs(node.tree, tree)
        self.assertIs(node.id, node_id)
        self.assertIs(node.name, node_name)

    def test_relational_data_immutable(self):
        node = Node(
            graph=MagicMock(),
            id=NodeID(0),
            name=NodeName("root"),
        )
        with self.assertRaises(AttributeError):
            node.tree = MagicMock()
        with self.assertRaises(AttributeError):
            node.id = NodeID(1)
        with self.assertRaises(AttributeError):
            node.name = NodeName("node")
