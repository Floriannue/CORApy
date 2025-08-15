"""
conZonotope - convert a polytope to a constrained zonotope (exact MATLAB translation)

Supports methods:
- 'exact:supportFunc' (default)
- 'exact:vertices'
Raises for fullspace/unbounded as in MATLAB.
"""

import numpy as np
from typing import Tuple

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.conZonotope.conZonotope import ConZonotope

from .private.priv_conZonotope_supportFunc import priv_conZonotope_supportFunc
from .private.priv_conZonotope_vertices import priv_conZonotope_vertices


def conZonotope(P, method: str = 'exact:supportFunc', B: Interval | None = None) -> ConZonotope:
    from .polytope import Polytope
    if not isinstance(P, Polytope):
        raise CORAerror('CORA:wrongInput', 'First argument must be a polytope')

    n = P.dim()

    # Default values
    if B is None:
        B = Interval(P)

    # fullspace cannot be represented as a conZonotope
    if P.representsa_('fullspace', 0):
        raise CORAerror('CORA:specialError', 'Polytope is unbounded and can therefore not be converted into a constrained zonotope.')

    # vertex representation given and polytope is empty
    if getattr(P, 'isVRep', False) and P.isVRep and (P.V.size == 0):
        return ConZonotope.empty(n)

    if method not in ['exact:vertices', 'exact:supportFunc']:
        raise CORAerror('CORA:wrongInput', 'method must be exact:vertices or exact:supportFunc')

    if method == 'exact:vertices':
        # calculate the vertices of the polytope (also check for unboundedness)
        try:
            V = P.vertices_()
        except Exception as e:
            if hasattr(P, '_bounded_val') and P._bounded_val is False:
                raise CORAerror('CORA:specialError', 'Polytope is unbounded and can therefore not be converted into a constrained zonotope.')
            raise
        # no vertices -> empty set
        if V.size == 0:
            return ConZonotope.empty(n)
        # ensure constraints
        P.constraints()
        c_, G_, A_, b_ = priv_conZonotope_vertices(P.A, P.b, P.Ae, P.be, V)
    else:
        # 'exact:supportFunc'
        if B is None:
            B = Interval(P)
        if getattr(P, '_bounded_val', None) is False:
            raise CORAerror('CORA:specialError', 'Polytope is unbounded and can therefore not be converted into a constrained zonotope.')
        if getattr(P, '_emptySet_val', None) is True:
            return ConZonotope.empty(n)
        P.constraints()
        c_, G_, A_, b_, empty = priv_conZonotope_supportFunc(P.A, P.b, P.Ae, P.be, B)
        if empty:
            return ConZonotope.empty(n)

    return ConZonotope(c_, G_, A_, b_)


