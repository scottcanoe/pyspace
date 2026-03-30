from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    NamedTuple,
    NewType,
    Protocol,
    Self,
    TypeVar,
)

import numpy as np
from bidict import bidict
from scipy.sparse import csgraph

NodeID = NewType("NodeID", str)
EdgeID = NewType("EdgeID", str)
UID = NewType("UID", int)


class GraphError(Exception):
    pass


class Node:
    """A node in the graph.

    Do not create these directly. They should always be created by a `Graph` instance.
    """

    def __init__(
        self,
        graph: Graph,
        node_id: NodeID,
        hash_value: int,
    ) -> None:
        self._graph = graph
        self._node_id = node_id
        self._hash_value = hash_value

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def node_id(self) -> NodeID:
        return self._node_id

    @property
    def data(self) -> Any:
        return self._graph._node_to_data["data"][self]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._node_id})"

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return self._hash_value


class Edge:
    def __init__(
        self,
        graph: Graph,
        edge_id: EdgeID,
        hash_value: int,
    ) -> None:
        self._graph = graph
        self._edge_id = edge_id
        self._hash_value = hash_value

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def edge_id(self) -> EdgeID:
        return self._edge_id

    @property
    def u(self) -> Node:
        return self._graph._edge_to_uv[self][0]

    @property
    def v(self) -> Node:
        return self._graph._edge_to_uv[self][1]

    @property
    def data(self) -> Any:
        return self._graph._edge_to_data[self]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._edge_id})"

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return self._hash_value



class PathStep(NamedTuple):
    edge: Edge
    inverse: bool


@dataclass
class GraphCache:
    node_index: bidict[Node, int]
    edge_directions: np.ndarray
    predecessors: np.ndarray


def auto_edge_id_from_nodes(u: Node, v: Node, sep: str = "->") -> EdgeID:
    return EdgeID(f"{u.node_id}{sep}{v.node_id}")


@dataclass(frozen=True)
class GraphToken:
    """Unique token for identifying theis graph instance. For hash stability."""
    _id: str


class Graph:
    def __init__(self) -> None:
        # Unique token for identifying theis graph instance. For hash stability.
        self._token = GraphToken(str(uuid.uuid4()))

        # Nodes
        self._node_id_generator = lambda: NodeID(str(uuid.uuid4()))
        self._nodes: bidict[NodeID, Node] = bidict()
        self._node_to_data: dict[NodeID, Any] = {}

        # Edges
        self._edge_id_generator: auto_edge_id_from_nodes
        self._edges: bidict[EdgeID, Edge] = bidict()
        self._edge_to_uv: bidict[Edge, tuple[Node, Node]] = bidict()
        self._edge_to_data: dict[Edge, Any] = {}

        # Path-finding
        self._cache: GraphCache | None = None

    @property
    def nodes(self) -> bidict[NodeID, Node]:
        return bidict(self._nodes)

    @property
    def edges(self) -> bidict[EdgeID, Edge]:
        return bidict(self._edges)

    @property
    def size(self) -> int:
        return len(self._nodes)

    def add_node(
        self,
        node_id: NodeID | None = None,
        data: Any | None = None,
    ) -> Node:
        """Add a node to the graph."""

        # Check `node_id` is avaialable.
        node_id = self._node_id_generator() if node_id is None else node_id
        if node_id in self._nodes:
            raise GraphError(f"Node ID {node_id} already in use.")

        hash_value = hash((self._token, node_id))
        node = Node(graph=self, node_id=node_id, hash_value=hash_value)
        self._nodes[node_id] = node
        self._node_to_data[node] = data
        self._cache = None
        return node

    def remove_node(self, node: Node | NodeID) -> None:
        node = self._as_valid_node(node)
        self._nodes.pop(node.node_id)
        self._node_to_data.pop(node.node)
        self._cache = None
        for uv, edge in self._edges.items():
            if node in uv:
                self.remove_edge(edge)
        self._cache = None

    def add_edge(
        self,
        u: Node | NodeID,
        v: Node | NodeID,
        edge_id: EdgeID | None = None,
        data: Any | None = None,
    ) -> Edge:
        u = self._as_valid_node(u)
        v = self._as_valid_node(v)
        if (u, v) in self._edge_to_uv.inv:
            raise GraphError(f"Edge from {u} --> {v} already exists")

        if (v, u) in self._edge_to_uv.inv:
            raise GraphError(f"Edge {u} <-- {v} already exists (inverse)")

        if edge_id is None:
            edge_id = auto_edge_id_from_nodes(u, v)
        if edge_id in self._edges:
            raise GraphError(f"Edge ID {edge_id} already in use.")

        hash_value = hash((self._token, edge_id))
        edge = Edge(graph=self, edge_id=edge_id, hash_value=hash_value)
        self._edges[edge_id] = edge
        self._edge_to_uv[edge] = (u, v)
        self._edge_to_data[edge] = data
        self._cache = None
        return edge

    def remove_edge(self, edge: Edge | EdgeID) -> None:
        edge = self._as_valid_edge(edge)
        self._edges.pop(edge.edge_id)
        self._edge_to_uv.pop(edge)
        self._edge_to_data.pop(edge)
        self._cache = None

    def clear_edges(self) -> None:
        self._edges.clear()
        self._edge_to_uv.clear()
        self._edge_to_data.clear()
        self._cache = None

    def shortest_path(
        self,
        source: Node | NodeID,
        target: Node | NodeID,
    ) -> list[PathStep]:
        """Return edges and edge directions for path between two nodes.

        Raises:
            RuntimeError: If start and end aren't connected.

        """
        if self._cache is None:
            self._compute_shortest_paths()

        source = self._as_valid_node(source)
        target = self._as_valid_node(target)
        if source == target:
            return []

        node_index: bidict[int, Node] = self._cache.node_index
        edge_directions: np.ndarray = self._cache.edge_directions
        predecessors: np.ndarray = self._cache.predecessors

        cur_node: Node = source
        cur_node_index = node_index.inv[cur_node]
        target_node_index = node_index.inv[target]

        if predecessors[target_node_index, cur_node_index] < 0:
            raise GraphError(f"No path found from {source} to {target}")

        path: list[PathStep] = []
        while cur_node_index != target_node_index:
            next_node_index = predecessors[target_node_index, cur_node_index]
            next_node = node_index[next_node_index]
            inverse = bool(edge_directions[next_node_index, cur_node_index] == -1)
            u, v = (next_node, cur_node) if inverse else (cur_node, next_node)
            edge = self._edge_to_uv.inv[(u, v)]
            step = PathStep(edge=edge, inverse=inverse)
            path.append(step)
            cur_node, cur_node_index = next_node, next_node_index

        return path

    def show(
        self,
        view: bool = False,
        filename: str = "graph",
        directory: os.PathLike | None = None,
        **kwargs,
    ):
        from pyspace.render import render_graph

        return render_graph(
            graph=self,
            view=view,
            filename=filename,
            directory=directory,
            **kwargs,
        )

    def _as_valid_node(self, node: Node | NodeID) -> Node:
        try:
            return self._nodes.get(node, self._nodes.inv.get(node))
        except KeyError:
            raise GraphError(f"{node} not found in graph")

    def _as_valid_edge(self, edge: Edge | EdgeID) -> Edge:
        try:
            return self._edges.get(edge, self._edges.inv.get(edge))
        except KeyError:
            raise GraphError(f"{edge} not found in graph")

    def _compute_shortest_paths(self) -> None:
        """Compute and store all shortest path info."""

        if len(self._nodes) == 0:
            self._cache = GraphCache(
                node_index=bidict(),
                edge_directions=np.zeros((0, 0), dtype=np.int8),
                predecessors=np.empty((0, 0), dtype=int),
            )
            return

        node_index = bidict({i: node for i, node in enumerate(self._nodes.values())})
        from_indices, to_indices = zip(
            *(
                (node_index.inv[edge.u], node_index.inv[edge.v])
                for edge in self._edges.values()
            )
        )
        edge_directions = np.zeros((len(self._nodes), len(self._nodes)), dtype=np.int8)
        edge_directions[to_indices, from_indices] = 1
        _, predecessors = csgraph.shortest_path(
            edge_directions,
            unweighted=True,
            directed=False,
            method="D",
            return_predecessors=True,
        )
        edge_directions[from_indices, to_indices] = -1

        self._cache = GraphCache(
            node_index=node_index,
            edge_directions=edge_directions,
            predecessors=predecessors,
        )

    def __getitem__(self, node_id: NodeID) -> Node:
        return self._nodes[node_id]


"""
--------------------------------------------------------------------------------------
Possible Constraints:
 - Unique parent
 - No cycles


"""


if __name__ == "__main__":
    # Create a graph with reference nodes A, B, and C.
    graph = Graph()
    root = graph.add_node("root")
    a = graph.add_node("a")
    b = graph.add_node("b")
    c = graph.add_node("c")
    x = graph.add_node("x")
    y = graph.add_node("y")
    z = graph.add_node("z")

    graph.add_edge("a", "root")
    graph.add_edge("b", "a")
    graph.add_edge("c", "a")
    graph.add_edge("x", "root")
    graph.add_edge("y", "x")
    graph.add_edge("z", "x")

    # graph.show(view=True)
    path = graph.shortest_path("a", "z")
    for step in path:
        print(step)
