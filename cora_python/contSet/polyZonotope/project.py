"""
project - projects a polynomial zonotope onto the specified dimensions

Syntax:
    pZ = project(pZ, dims)

Inputs:
    pZ - (polyZonotope) polynomial zonotope
    dims - dimensions for projection (0-based indices)

Outputs:
    pZ - (polyZonotope) projected polynomial zonotope

Example:
    pZ = PolyZonotope(c, G, GI, E, id)
    pZ_01 = project(pZ, [0, 1])
    pZ_02 = project(pZ, [0, 2])

See also: zonotope/project

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       25-June-2018 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope


def project(pZ: 'PolyZonotope', dims: Union[List[int], np.ndarray]) -> 'PolyZonotope':
    """
    Projects a polynomial zonotope onto the specified dimensions.

    MATLAB: pZ.c = pZ.c(dims,:); pZ.G = pZ.G(dims,:); pZ.GI = pZ.GI(dims,:);
    Python uses 0-based dims.
    """
    from .polyZonotope import PolyZonotope

    dims = np.atleast_1d(np.asarray(dims, dtype=int))

    c_proj = pZ.c[dims, :]
    G_proj = pZ.G[dims, :] if pZ.G.size > 0 else pZ.G
    GI_proj = None if pZ.GI is None else pZ.GI[dims, :]

    return PolyZonotope(c_proj, G_proj, GI_proj, pZ.E, pZ.id)
