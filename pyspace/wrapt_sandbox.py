"""Wrap any object and attach extra callables or values visible as attributes.

Built on :class:`wrapt.ObjectProxy`, so operators, iteration, and most dunders
still target the inner instance. Extension names are resolved before delegating
to the wrapped object, so they can shadow attributes on it.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import wrapt


@dataclass(frozen=True)
class NodeMeta:
    tree: Any
    uid: int
    node_id: int
    node_name: str | None



_META_FIELD = "_self_node"


class NodeProxy(wrapt.ObjectProxy):
    """Transparent proxy with optional extra attributes and bound methods.

    Callables in ``**extensions`` are bound to the proxy (first parameter is
    the proxy, so use ``self.__wrapped__`` when you need the bare instance).
    """

    def __init__(self, wrapped: Any, node_meta: NodeMeta) -> None:
        super().__init__(wrapped)
        object.__setattr__(self, _META_FIELD, node_meta)

    def __getattr__(self, name: str) -> Any:
        node_meta: NodeMeta = object.__getattribute__(self, _META_FIELD)
        if hasattr(node_meta, name):
            member = getattr(node_meta, name)
            if callable(member):
                return types.MethodType(member, self)
            return member
        return super().__getattr__(name)

    # def extend(self, **extensions: Any) -> NodeProxy:
    #     """Register more extensions; returns ``self`` for chaining."""
    #     object.__getattribute__(self, _META_FIELD).update(extensions)
    #     return self


def wrap(wrapped: Any, node_meta: NodeMeta) -> NodeProxy:
    """Construct an :class:`ExtendedProxy` around ``wrapped``."""
    return NodeProxy(wrapped, node_meta)


def get_node_meta(node: NodeProxy) -> NodeMeta:
    return node.__getattribute__(_META_FIELD)

def set_node_meta(obj: Any, node_meta: NodeMeta) -> None:
    object.__setattr__(obj, _META_FIELD, node_meta)

def has_node_meta(obj: Any) -> bool:
    return hasattr(obj, _META_FIELD)


class Node:
    def __init__(self) -> None:
        pass
        
a = Node()

a_meta = NodeMeta(tree=None, uid=1234, node_id=0, node_name="a")
A = NodeProxy(a, a_meta)


