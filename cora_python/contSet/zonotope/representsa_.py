import numpy as np
from typing import Tuple, Union, Optional, Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


"""
representsa_ - checks if a zonotope can also be represented by a
    different set, e.g., a special case

Syntax:
    res = representsa_(Z,type,tol)
    [res,S] = representsa_(Z,type,tol)

Inputs:
    Z - zonotope object
    type - other set representation or 'origin', 'point', 'hyperplane'
    tol - tolerance

Outputs:
    res - true/false
    S - converted set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/representsa

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       19-July-2023 (MATLAB)
Last update:   28-May-2025 (TL, speed up for 'interval') (MATLAB)
             2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Tuple, Union, Optional, Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def representsa_(Z, set_type: str, tol: float = 1e-12, method: str = 'linearize', iter_val: int = 1, splits: int = 0, **kwargs):
    # Import here to avoid circular imports
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.capsule import Capsule
    
    # Check if we need to return the converted set
    return_set = 'return_set' in kwargs and kwargs['return_set']

    # Handle empty list input (e.g., missing set -> empty)
    if isinstance(Z, list) and len(Z) == 0:
        res = (set_type == 'emptySet')
        if return_set:
            return res, Z
        return res
    
    # Check empty object case
    try:
        from cora_python.contSet.contSet.representsa_emptyObject import representsa_emptyObject
        if return_set:
            empty, res, S = representsa_emptyObject(Z, set_type)
            if empty:
                return res, S
        else:
            empty, res = representsa_emptyObject(Z, set_type)
            if empty:
                return res
    except:
        # If representsa_emptyObject not available, continue
        pass
    
    # Dimension
    n = Z.dim()
    
    # Init second output argument (covering all cases with res = false)
    S = None
    
    # Delete zero-length generators
    Z_compact = Z.compact_('zeros', np.finfo(float).eps)
    # Number of generators
    nrGens = Z_compact.G.shape[1] if Z_compact.G.size > 0 else 0
    
    if set_type == 'origin':
        c = Z_compact.c
        res = (c is not None and c.size > 0 and 
               ((nrGens == 0 and np.allclose(c, 0, atol=tol)) or 
                Z_compact.interval().norm_(2) <= tol))
        if return_set and res:
            S = np.zeros((n, 1))
    
    elif set_type == 'point':
        res = nrGens == 0 or (Z_compact - Z_compact.c).interval().norm_(2) <= tol
        if return_set and res:
            S = Z_compact.c
    
    elif set_type == 'capsule':
        # True if only one or less generators
        res = n == 1 or nrGens <= 1
        if return_set and res:
            S = Capsule(Z_compact.c, Z_compact.G, 0)
    
    elif set_type == 'conHyperplane':
        # TODO: condition?
        raise CORAerror('CORA:notSupported',
                       f'Comparison of zonotope to {set_type} not supported.')
    
    elif set_type == 'conPolyZono':
        res = True
        if return_set:
            # Convert zonotope to conPolyZono via polyZonotope
            from ..conPolyZono import ConPolyZono
            from ..polyZonotope import PolyZonotope
            pZ = PolyZonotope(Z_compact)
            S = ConPolyZono(pZ)
    
    elif set_type == 'conZonotope':
        res = True
        if return_set:
            # Convert zonotope to conZonotope using constructor
            from ..conZonotope import ConZonotope
            S = ConZonotope(Z_compact.c, Z_compact.G)
    
    elif set_type == 'ellipsoid':
        raise CORAerror('CORA:notSupported',
                       f'Comparison of zonotope to {set_type} not supported.')
    
    elif set_type == 'halfspace':
        # Zonotopes cannot be unbounded
        res = False
    
    elif set_type == 'interval':
        res = _aux_isInterval(Z_compact, tol)
        if return_set and res:
            # Get interval and ensure it's in the expected format (column vectors)
            I = Z_compact.interval()
            # Convert to column vector format if needed
            if I.inf.ndim == 1:
                I.inf = I.inf.reshape(-1, 1)
                I.sup = I.sup.reshape(-1, 1)
            S = I
    
    elif set_type == 'levelSet':
        res = True
        if return_set:  # no conversion
            raise CORAerror('CORA:notSupported',
                           'Conversion from zonotope to levelSet not supported.')
    
    elif set_type == 'polytope':
        res = True
        if return_set:
            # Convert zonotope to polytope using exact method
            from ..polytope import Polytope
            S = Z_compact.polytope('exact')
    
    elif set_type == 'polyZonotope':
        res = True
        if return_set:
            # Convert zonotope to polyZonotope
            from ..polyZonotope import PolyZonotope
            c = Z_compact.c
            G = Z_compact.G
            E = np.eye(G.shape[1]) if G.size > 0 else np.empty((0, 0))
            GI = np.empty((c.shape[0], 0))  # Empty independent generators
            S = PolyZonotope(c, G, GI, E)
    
    elif set_type == 'probZonotope':
        # Is never true
        res = False
    
    elif set_type == 'zonoBundle':
        res = True
        if return_set:
            # Convert zonotope to zonoBundle
            from ..zonoBundle import ZonoBundle
            S = ZonoBundle([Z_compact])
    
    elif set_type == 'zonotope':
        # Obviously true
        res = True
        if return_set:
            S = Z_compact
    
    elif set_type == 'hyperplane':
        # Zonotopes cannot be unbounded
        res = False
    
    elif set_type == 'parallelotope':
        res = n == 1 or _aux_isParallelotope(Z_compact, tol)
        if return_set and res:
            S = Z_compact
    
    elif set_type == 'convexSet':
        res = True
    
    elif set_type == 'emptySet':
        # Check if zonotope represents empty set
        res = Z_compact.isemptyobject()
    
    elif set_type == 'fullspace':
        # Zonotope cannot be unbounded
        res = False
    
    else:
        raise CORAerror('CORA:notSupported',
                       f'Comparison of zonotope to {set_type} not supported.')
    
    if return_set:
        return res, S
    else:
        return res


# Auxiliary functions -----------------------------------------------------

def _aux_isInterval(Z, tol):
    """
    Check if zonotope is axis-aligned (can be represented as interval)
    """
    res = True
    # One-dimensional zonotopes are always intervals
    if Z.dim() == 1:
        return res
    
    # Check if no generator has more than one entry
    G = Z.G
    if G.size == 0:
        return res
    
    # Check each generator
    for i in range(G.shape[1]):
        gen = G[:, i]
        # Count non-zero entries (within tolerance)
        non_zero_count = np.sum(np.abs(gen) > tol)
        if non_zero_count > 1:
            res = False
            break
    
    return res


def _aux_isParallelotope(Z, tol):
    """
    Check if zonotope is a parallelotope
    """
    res = True
    
    # Dimension and generators of zonotope
    n = Z.dim()
    G = Z.G
    
    # One-dimensional zonotopes are always parallelotopes (with at least one generator)
    if n == 1 and G.shape[1] > 0:
        return res
    
    # Delete zero-length generators
    Z_compact = Z.compact_('zeros', np.finfo(float).eps)
    G = Z_compact.G
    
    # Quick check: not enough generators
    if G.shape[1] < n:
        res = False
        return res
    
    if Z_compact.isFullDim() and G.shape[1] == n:
        return res
    
    # Not a parallelotope
    res = False
    return res 