"""
plot3D - plots a 3D projection of a contSet

Syntax:
   han = plot3D(S)
   han = plot3D(S,dims)
   han = plot3D(S,dims,NVpairs)

Authors:       Tobias Ladner
Written:       14-October-2024
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def plot3D(E: Ellipsoid, plot_kwargs: Optional[Dict[str, Any]] = None, nvpairs_vertices: Optional[List[Any]] = None):
    # default params
    if plot_kwargs is None:
        plot_kwargs = {}
    if nvpairs_vertices is None:
        nvpairs_vertices = []

    # outer-approximate using zonotope
    Z = E.zonotope('outer:norm', 100)

    # plot zonotope enclosure (delegates to base contSet plot3D via dynamic dispatch)
    from cora_python.contSet.contSet.plot3D import plot3D as plot3D_base
    return plot3D_base(Z, plot_kwargs, nvpairs_vertices)

