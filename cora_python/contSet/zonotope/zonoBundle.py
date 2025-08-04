"""
zonoBundle - convert a zonotope object into a zonotope bundle object

Syntax:
    zB = zonoBundle(Z)

Inputs:
    Z - zonotope object

Outputs:
    zB - zonoBundle object

Example:
    from cora_python.contSet.zonotope import Zonotope, zonoBundle
    import numpy as np
    Z = Zonotope(np.array([[0], [0]]), np.eye(2))
    zB = zonoBundle(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: polytope/zonoBundle

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       26-November-2019 (MATLAB)
Last update:   --- (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
from .zonotope import Zonotope
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonoBundle import ZonoBundle

def zonoBundle(Z: Zonotope) -> 'ZonoBundle':
    """
    Convert a zonotope object into a zonotope bundle object.
    """
    from ..zonoBundle import ZonoBundle
    
    # Create zonoBundle with a single zonotope
    zB = ZonoBundle([Z])
    
    return zB 