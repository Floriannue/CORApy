"""
representsa_ - checks if a zonoBundle can also be represented by a different set,
    e.g., a special case

Syntax:
    res = representsa_(zB, type, tol)
    res, S = representsa_(zB, type, tol)

Inputs:
    zB - zonoBundle object
    type - other set representation or 'origin', 'point', 'hyperplane'
    tol - tolerance

Outputs:
    res - true/false
    S - converted set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/representsa

Authors:       Mark Wetzlinger, Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       19-July-2023 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Tuple, Optional, Any

if TYPE_CHECKING:
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle

from cora_python.contSet.contSet.representsa_emptyObject import representsa_emptyObject
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def representsa_(zB: 'ZonoBundle', set_type: str, tol: float = 1e-12, **kwargs) -> Union[bool, Tuple[bool, Optional[Any]]]:
    """
    Checks if a zonoBundle can also be represented by a different set type.
    
    Args:
        zB: zonoBundle object
        set_type: other set representation or 'origin', 'point', 'hyperplane'
        tol: tolerance
        **kwargs: additional arguments (e.g., return_set)
    
    Returns:
        bool or tuple: Whether zB can be represented by set_type, optionally with converted set
    """
    # Check empty object case
    return_set = kwargs.get('return_set', False)
    if return_set:
        empty, res, S = representsa_emptyObject(zB, set_type)
    else:
        empty, res, _ = representsa_emptyObject(zB, set_type)
        S = None
    if empty:
        if return_set:
            return res, S
        return res
    
    # dimension
    n = zB.dim()
    
    # init second output argument (covering all cases with res = false)
    S = None
    
    if set_type == 'origin':
        cZ = _conZonotope(zB)
        if hasattr(cZ, 'representsa_'):
            res = cZ.representsa_('origin', tol)
        else:
            from cora_python.contSet.contSet.representsa import representsa
            res = representsa(cZ, 'origin', tol)
        if return_set and res:
            # return origin as zero vector
            S = np.zeros((n, 1))
    
    elif set_type == 'conPolyZono':
        res = True
        if return_set:
            S = _conPolyZono(zB)
    
    elif set_type == 'conZonotope':
        res = True
        if return_set:
            S = _conZonotope(zB)
    
    elif set_type == 'halfspace':
        # zonoBundles are bounded
        res = False
    
    elif set_type == 'interval':
        res = (n == 1 or _aux_isInterval(zB, tol))
        if return_set and res:
            S = zB.interval()
    
    elif set_type == 'levelSet':
        res = True
        if return_set:
            # not yet supported
            raise CORAerror('CORA:notSupported',
                          f'Conversion from zonoBundle to {set_type} not supported.')
    
    elif set_type == 'polytope':
        res = True
        if return_set:
            S = _polytope(zB)
    
    elif set_type == 'polyZonotope':
        res = True
        if return_set:
            S = _polyZonotope(zB)
    
    elif set_type == 'probZonotope':
        # cannot be true
        res = False
    
    elif set_type == 'zonoBundle':
        # obviously true
        res = True
        if return_set:
            S = zB
    
    elif set_type == 'hyperplane':
        # zonotope bundles are bounded
        res = False
    
    elif set_type == 'convexSet':
        res = True
    
    elif set_type == 'emptySet':
        res = _aux_emptySet(zB, tol)
        if return_set and res:
            from cora_python.contSet.emptySet.emptySet import EmptySet
            S = EmptySet(n)
    
    elif set_type in ['point', 'capsule', 'conHyperplane', 'ellipsoid', 'zonotope', 'parallelotope']:
        raise CORAerror('CORA:notSupported',
                       f'Comparison of zonoBundle to {set_type} not supported.')
    
    elif set_type == 'fullspace':
        # zonotope bundles are bounded
        res = False
    
    else:
        # Unknown set type
        res = False
    
    if return_set:
        return res, S
    return res


def _aux_isInterval(zB: 'ZonoBundle', tol: float) -> bool:
    """
    Auxiliary function to check if zonoBundle is an interval
    
    Args:
        zB: zonoBundle object
        tol: tolerance
    
    Returns:
        bool: True if zB can be represented as an interval
    """
    # one-dimensional zonoBundles are always intervals
    if zB.dim() == 1:
        return True
    
    # empty sets can also be represented by intervals
    if representsa_(zB, 'emptySet', tol):
        return True
    
    # all other cases: check individual zonotopes
    # (note: there are cases where the intersection is still an interval)
    for i in range(zB.parallelSets):
        if hasattr(zB.Z[i], 'representsa_'):
            if not zB.Z[i].representsa_('interval', tol):
                return False
        else:
            from cora_python.contSet.contSet.representsa import representsa
            if not representsa(zB.Z[i], 'interval', tol):
                return False
    
    return True


def _aux_emptySet(zB: 'ZonoBundle', tol: float) -> bool:
    """
    Auxiliary function to check if zonoBundle is empty
    
    Args:
        zB: zonoBundle object
        tol: tolerance
    
    Returns:
        bool: True if zB is empty
    """
    # full-empty zonotope bundle
    if zB.parallelSets == 0:
        return True
    
    # check if any of the single zonotopes is empty
    for i in range(len(zB.Z)):
        if hasattr(zB.Z[i], 'representsa_'):
            if zB.Z[i].representsa_('emptySet', np.finfo(float).eps):
                return True
        else:
            from cora_python.contSet.contSet.representsa import representsa
            if representsa(zB.Z[i], 'emptySet', np.finfo(float).eps):
                return True
    
    # check if the intersection of the zonotopes is empty
    cZ = _conZonotope(zB)
    if hasattr(cZ, 'representsa_'):
        return cZ.representsa_('emptySet', tol)
    else:
        from cora_python.contSet.contSet.representsa import representsa
        return representsa(cZ, 'emptySet', tol)


def _conZonotope(zB: 'ZonoBundle'):
    """
    Convert zonoBundle to conZonotope (helper function)
    
    Args:
        zB: zonoBundle object
    
    Returns:
        ConZonotope object
    """
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    
    if zB.parallelSets == 0:
        return ConZonotope.empty(zB.dim())
    
    # initialization
    cZ = ConZonotope(zB.Z[0])
    
    # calculate the intersection of the parallel sets
    for i in range(1, zB.parallelSets):
        temp = ConZonotope(zB.Z[i])
        cZ = cZ.and_(temp, 'exact')
    
    return cZ


def _polytope(zB: 'ZonoBundle', *varargin):
    """
    Convert zonoBundle to polytope (helper function)
    
    Args:
        zB: zonoBundle object
        *varargin: optional method arguments
    
    Returns:
        Polytope object
    """
    from cora_python.contSet.polytope.polytope import Polytope
    
    if zB.parallelSets == 0:
        return Polytope.empty(zB.dim())
    
    # compute over-approximative polytope for each zonotope
    Ptmp = []
    for i in range(zB.parallelSets):
        if hasattr(zB.Z[i], 'polytope'):
            Ptmp.append(zB.Z[i].polytope(*varargin))
        else:
            from cora_python.contSet.zonotope.polytope import polytope
            Ptmp.append(polytope(zB.Z[i], *varargin))
    
    # intersect all polytopes
    P = Ptmp[0]
    for i in range(1, zB.parallelSets):
        P = P.and_(Ptmp[i], 'exact')
    
    # set properties
    if hasattr(P, 'bounded'):
        P.bounded = True
    
    return P


def _polyZonotope(zB: 'ZonoBundle'):
    """
    Convert zonoBundle to polyZonotope (helper function)
    
    Args:
        zB: zonoBundle object
    
    Returns:
        PolyZonotope object
    """
    from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
    
    poly = _polytope(zB)
    if hasattr(poly, 'polyZonotope'):
        pZ = poly.polyZonotope()
    else:
        from cora_python.contSet.polytope.polyZonotope import polyZonotope
        pZ = polyZonotope(poly)
    
    return pZ


def _conPolyZono(zB: 'ZonoBundle'):
    """
    Convert zonoBundle to conPolyZono (helper function)
    
    Args:
        zB: zonoBundle object
    
    Returns:
        ConPolyZono object
    """
    from cora_python.contSet.conPolyZono.conPolyZono import ConPolyZono
    
    cZ = _conZonotope(zB)
    if hasattr(cZ, 'conPolyZono'):
        cPZ = cZ.conPolyZono()
    else:
        from cora_python.contSet.conZonotope.conPolyZono import conPolyZono
        cPZ = conPolyZono(cZ)
    
    return cPZ

