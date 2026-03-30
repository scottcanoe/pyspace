"""Graphviz rendering for a FrameGraph."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from graphviz import Digraph

if TYPE_CHECKING:
    from pyspace.graph import FrameGraph

DEFAULT_GRAPHVIZ_DIRECTORY = Path(__file__).parent.parent / "local" / "graphviz"


def render_graph(
    graph: FrameGraph,
    filename: str = "graph",
    directory: os.PathLike | None = None,
    view: bool = False,
    **kwargs,
) -> Digraph:
    if directory is None:
        directory = DEFAULT_GRAPHVIZ_DIRECTORY
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    g = Digraph()
    for frame_id in graph.frames:
        g.node(str(frame_id))
    for (from_frame, to_frame) in graph.transforms:
        g.edge(str(from_frame.frame_id), str(to_frame.frame_id))
    g.render(filename=filename, directory=directory, **kwargs)

    if view:
        g.view()

    return g
