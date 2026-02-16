"""
polygon - creates a polygon enclosure of a 2-dimensional polyZonotope

Syntax:
    pgon = polygon(pZ)
    pgon = polygon(pZ, splits)

Inputs:
    pZ - polyZonotope object
    splits - number of splits for refinement (optional, default 8)

Outputs:
    pgon - polygon object

Example:
    pZ = polyZonotope([0;0],[2 0 1;0 2 1],[0;0],[1 0 3;0 1 1]);
    pgon = polygon(pZ, 8);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: plot

Authors:       Niklas Kochdumper (MATLAB)
               Python translation: match MATLAB behavior (zonotope over-approximation)
Written:       08-April-2020 (MATLAB)
Last update:   29-June-2024 (TL, MATLAB bug fix)
Python:        PolyZonotope-specific polygon so plot path uses splits, not vertices(method).
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def polygon(pZ: 'PolyZonotope', splits: int = 8):
    """
    Creates a polygon enclosure of a 2-dimensional polyZonotope.

    Mirrors MATLAB: polygon(pZ, splits) uses splits for refinement and does NOT
    call vertices(pZ, method). Instead it uses zonotope over-approximation and
    vertices of the zonotope. This avoids the contract where plot passes
    nvpairs_polygon = [8] (splits) and the generic polygon would call
    vertices(pZ, 8), which fails because vertices expects a method string.

    Args:
        pZ: polyZonotope object
        splits: number of splits for refinement (optional, default 8).
                Currently only the zonotope over-approximation is used;
                the full MATLAB split loop can be added later.

    Returns:
        pgon: polygon object
    """
    from cora_python.contSet.polygon.polygon import Polygon
    from .dim import dim
    from .zonotope import zonotope
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.zonotope.vertices_ import vertices_

    eps = np.finfo(float).eps

    # Quick return if pZ is empty
    if pZ.representsa_('emptySet', eps):
        return Polygon()

    # Quick return if pZ represents a zonotope: use Z's polygon
    try:
        res, Z = pZ.representsa_('zonotope', eps, return_set=True)
    except TypeError:
        res = pZ.representsa_('zonotope', eps)
        Z = None
    if res and Z is not None:
        from cora_python.contSet.contSet.polygon import polygon as contset_polygon
        return contset_polygon(Z)

    # Only 2D supported
    if dim(pZ) != 2:
        raise CORAerror('CORA:noExactAlg', pZ,
            'Method "polygon" is only applicable for 2D polyZonotopes!')

    # Zonotope over-approximation, then vertices of zonotope, then polygon
    # (MATLAB aux_getPolygon: Z = zonotope(pZ), V = vertices_(Z), poly = polyshape(V', ...))
    Z = zonotope(pZ)
    V = vertices_(Z)
    if V.size == 0:
        return Polygon()
    pgon = Polygon(V)
    return pgon
