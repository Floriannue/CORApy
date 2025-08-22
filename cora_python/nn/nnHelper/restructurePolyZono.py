"""
restructurePolyZono - Calculate a new over-approxmiating representation 
   of a polynomial zonotope in such a way that there remain no 
   independent generators

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Tuple


def restructurePolyZono(c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray, 
                        id_: np.ndarray, id_max: int, nrGen: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate a new over-approximating representation of a polynomial zonotope.
    
    Args:
        c: center of polyZonotope
        G: dep. generator of polyZonotope
        GI: indep. generator of polyZonotope
        E: exponential matrix of polyZonotope
        id_: ids
        id_max: max id
        nrGen: number of generators
        
    Returns:
        c, G, GI, E, id_: restructured polynomial zonotope
        
    See also: -
    """
    # restructure
    from cora_python.contSet.polyZonotope import PolyZonotope
    
    pZ = PolyZonotope(c, G, GI, E, id_)
    pZ = pZ.restructure('reduceGirard', nrGen / len(c))
    
    # read properties
    c = pZ.c
    G = pZ.G
    GI = pZ.GI
    E = pZ.E
    id_ = pZ.id
    
    return c, G, GI, E, id_
