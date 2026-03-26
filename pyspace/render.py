import os
from pathlib import Path
from typing import TYPE_CHECKING

from graphviz import Digraph

if TYPE_CHECKING:
    from pyspace.graph import Graph


DEFAULT_GRAPHVIZ_DIRECTORY = Path(__file__).parent.parent / "local" / "graphviz"


def render_graph(
    graph: Graph,
    filename: str = "graph",
    directory: os.PathLike | None = None,
    view: bool = False,
    **kwargs) -> None:

    if directory is None:
        directory = DEFAULT_GRAPHVIZ_DIRECTORY
    g = Digraph()
    for frame in graph.frames:
        g.node(frame.frame_id)
    for transform in graph.transforms:
        g.edge(transform.from_frame.frame_id, transform.to_frame.frame_id)
    g.render(filename=filename, directory=directory, **kwargs)

    if view:
        g.view()
    
    return g

