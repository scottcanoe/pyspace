"""Tree of nodes backed by a sparse adjacency matrix (pytransform3d-style).

Relationships are stored as directed parent→child edges in parallel index lists
``i`` (parent column index) and ``j`` (child column index), then folded into a
CSR matrix ``connections`` for graph algorithms — mirroring
:class:`pytransform3d.transform_manager.TransformGraphBase`.
"""

from __future__ import annotations

import types
import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, NamedTuple, NewType, Protocol

import bidict
import numpy as np
import wrapt
from scipy.sparse import csgraph, csr_matrix

__all__ = [
    "Node",
    "NodeID",
    "Graph",
]

NodeID = NewType("NodeID", int)
NodeName = NewType("NodeName", str)



class TreeError(Exception):
    pass


@dataclass(frozen=True)
class Node:
    """A Node in a :class:`NodeTree`.

    Parent/child queries always go through the owning tree's adjacency data.
    """
    graph: Graph
    id: NodeID
    name: NodeName | None
    data: Any | None
    def __repr__(self) -> str:
        return f"Node(id={self.id}, name={self.name})"


@dataclass(frozen=True)
class Edge:
    from_node: Node
    to_node: Node


@dataclass(frozen=True)
class PathStep:
    from_node: Node
    to_node: Node
    invert: bool


class Graph:

    def __init__(self) -> None:
        # nodes
        self._next_node_id: int = 0  # A node ID counter.
        self._root: Node | None = None
        self._id_to_node: dict[NodeID, Node] = {}
        self._name_to_node: dict[NodeName, Node] = {}
        # edges
        self._edges: list[Edge] = []
        # paths
        self._refresh_paths()

    @property
    def root(self) -> Node | None:
        return self._root

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._id_to_node.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges)
    
    @property
    def size(self) -> int:
        return len(self._id_to_node)
        
    def create_node(
        self,
        name: str | None = None,
        data: Any | None = None,
        ) -> Node:
        """Create a new node and add it to the graph."""

        if name is not None and name in self._name_to_node:
            raise TreeError(f"name {name} already in use.")

        node = Node(
            self,
            id=self._next_node_id,
            name=name,
            data=data,
        )
        self._next_node_id += 1
        self._id_to_node[node.id] = node
        if name is not None:
            self._name_to_node[node.name] = node
        self._is_stale = True
        return node

    def create_edge(self, from_node: Node, to_node: Node) -> None:
        edge = Edge(from_node=from_node, to_node=to_node)
        self._edges.append(edge)
        self._is_stale = True

    def shortest_path(self, start: Node, end: Node) -> tuple[PathStep]:
        """Return nodes and inverse flags for path from start to end nodes.
        
        Returned nodes don't include the start node.

        Raises:
            RuntimeError: If start and end aren't connected.
        
        """
        if self._is_stale:
            self._refresh_paths()
        if self._predecessors[end.id, start.id] < 0:
            raise TreeError(f"No path found from {start} to {end}")
        
        from_node: Node = start
        steps: list[PathStep] = []
        while from_node.id != end.id:
            to_node = self._id_to_node[self._predecessors[end.id, from_node.id]]
            invert = bool(self._adjacency_matrix[to_node.id, from_node.id] == 0)
            steps.append(PathStep(from_node=from_node, to_node=to_node, invert=invert))
            from_node = to_node
        return tuple(steps)

    def _refresh_paths(self) -> None:
        if self.size == 0:
            self._adjacency_matrix = csr_matrix((0, 0))
            self._path_lengths = np.empty((0, 0), dtype=int)
            self._predecessors = np.empty((0, 0), dtype=int)
            self._is_stale = False
            return
        
        rank = self._next_node_id
        from_nodes = [edge.from_node.id for edge in self._edges]
        to_nodes = [edge.to_node.id for edge in self._edges]
        vals = np.ones(len(from_nodes), dtype=int)
        self._adjacency_matrix = csr_matrix(
            ((vals, (to_nodes, from_nodes))), shape=(rank, rank)
        )
        self._path_lengths, self._predecessors = csgraph.shortest_path(
            self._adjacency_matrix,
            unweighted=True,
            directed=False,
            method="D",
            return_predecessors=True,
        )
        self._is_stale = False

    def __contains__(self, node: Any) -> bool:
        return node in self._id_to_node.values()

    def __getitem__(self, name: NodeName) -> Node:
        return self._name_to_node[name]


@dataclass(frozen=True)
class NodeMeta:
    tree: Any
    uid: int
    node_id: int
    node_name: str | None

_NODE_META_FIELD = "_self_node"

class NodeProxy(wrapt.ObjectProxy):
    """Transparent proxy with optional extra attributes and bound methods.

    Callables in ``**extensions`` are bound to the proxy (first parameter is
    the proxy, so use ``self.__wrapped__`` when you need the bare instance).
    """

    def __init__(self, wrapped: Any, node_meta: NodeMeta) -> None:
        if hasattr(wrapped, _NODE_META_FIELD):
            raise NotImplementedError('what do when already wrapped?')
        super().__init__(wrapped)
        object.__setattr__(self, _NODE_META_FIELD, node_meta)

    @property
    def node(self):
        return object.__getattribute__(self, _NODE_META_FIELD)

    # def add_node(self, obj: Any) -> NodeProxy:
    #     # wrap child if not a node/nodeproxy

    #     node_meta: NodeMeta = object.__getattribute__(self, _NODE_META_FIELD)
    #     return node_meta.tree.add_child(self, child)

    def __getattr__(self, name: str) -> Any:
        node_meta: NodeMeta = object.__getattribute__(self, _NODE_META_FIELD)
        if hasattr(node_meta, name):
            member = getattr(node_meta, name)
            if callable(member):
                return types.MethodType(member, self)
            return member
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        node_meta: NodeMeta = object.__getattribute__(self, _NODE_META_FIELD)
        if name in node_meta.__dataclass_fields__:
            raise AttributeError(f"NodeMeta field {name} is read-only")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        node_meta: NodeMeta = object.__getattribute__(self, _NODE_META_FIELD)
        if name in node_meta.__dataclass_fields__:
            raise AttributeError(f"NodeMeta field {name} is read-only")
        super().__delattr__(name)

    def __repr__(self) -> str:
        return repr(self.__wrapped__)


def render_tree(tree: Graph, view: bool = True) -> None:
    from graphviz import Digraph
    g = Digraph()
    for node in tree.nodes:
        g.node(node.name)
    for edge in tree._edges:
        from_node = edge.from_node
        to_node = edge.to_node
        g.edge(from_node.name, to_node.name)
    
    return g



tree = Graph()
root = tree.create_node(name="root")  # 0
a = tree.create_node(name="a")        # 1
b = tree.create_node(name="b")        # 2
c = tree.create_node(name="c")        # 3
x = tree.create_node(name="x")        # 4
y = tree.create_node(name="y")        # 5
z = tree.create_node(name="z")        # 6
# d = tree.create_node(name="d")
# e = tree.create_node(name="e")
# z = tree.create_node(name="z")
tree.create_edge(a, root)
tree.create_edge(b, a)
tree.create_edge(c, b)
tree.create_edge(x, root)
tree.create_edge(y, x)
tree.create_edge(z, y)
tree._refresh_paths()

print("adjacency matrix:")
print(tree._adjacency_matrix.todense())
print("predecessors:")
print(tree._predecessors)

g = render_tree(tree, view=False)

self = tree
if self._is_stale:
    self._refresh_paths()

start = z
end = b
steps = tree.shortest_path(start, end)
print("shortest path steps:")
for step in steps:
    print(step)
