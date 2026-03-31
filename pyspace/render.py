import os
from pathlib import Path
from typing import TYPE_CHECKING

from graphviz import Digraph

if TYPE_CHECKING:
    from pyspace.transform_graph import Graph


DEFAULT_GRAPHVIZ_DIRECTORY = Path(__file__).parent.parent / "local" / "graphviz"


def render_graph(
    graph,
    filename: str = "graph",
    directory: os.PathLike | None = None,
    view: bool = False,
    **kwargs,
) -> None:

    if directory is None:
        directory = DEFAULT_GRAPHVIZ_DIRECTORY
    g = Digraph()
    for node_id, node in graph.nodes.items():
        g.node(node_id)
    for edge_id, edge in graph.edges.items():
        g.edge(edge.u.node_id, edge.v.node_id)
    g.render(filename=filename, directory=directory, **kwargs)

    if view:
        g.view()
    
    return g

