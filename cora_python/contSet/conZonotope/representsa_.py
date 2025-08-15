"""
representsa_ - checks if a constrained zonotope can also be represented
    by a different set, e.g., a special case

Syntax:
    res = representsa_(cZ,type,tol)
    [res,S] = representsa_(cZ,type,tol)

Inputs:
    cZ - ConZonotope object
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
Last update:   ---
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

from scipy.linalg import null_space
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog

if TYPE_CHECKING:
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    from cora_python.contSet.contSet import ContSet


def representsa_(cZ: 'ConZonotope', set_type: str, tol: float = 1e-12, **kwargs) -> Union[bool, Tuple[bool, 'ContSet']]:
    """
    Checks if a constrained zonotope can also be represented by a different set type.

    This function supports two calling patterns:
    - res = representsa_(cZ, set_type)  -> returns bool
    - res, S = representsa_(cZ, set_type)  -> returns (bool, converted_set)

    Args:
        cZ: ConZonotope object
        set_type: other set representation or 'origin', 'point', 'hyperplane'
        tol: tolerance (default: 1e-12)

    Returns:
        bool or tuple: Whether cZ can be represented by set_type, optionally with converted set
    """
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.zonotope import Zonotope
    
    def _return_result(res_val, set_val=None):
        if kwargs.get('return_set', False):
            return res_val, set_val
        return res_val

    # check empty object case
    empty_result = cZ.representsa_emptyObject(set_type)
    if isinstance(empty_result, tuple) and len(empty_result) == 3:
        empty, res, S = empty_result
        if empty:
            return _return_result(res, S)
    elif isinstance(empty_result, tuple) and len(empty_result) == 2:
        empty, res = empty_result
        if empty:
            return _return_result(res, None)

    # dimension
    n = cZ.dim()

    # init second output argument (covering all cases with res = false)
    S = None

    if set_type == 'origin':
        Z = Zonotope(cZ.c, cZ.G)
        I = Interval(Z)
        if hasattr(I, 'representsa_'):
            res = I.representsa_('origin', tol)
        else:
            res = I.representsa_('origin', tol)
        if res and kwargs.get('return_set', False):
            S = np.zeros((n, 1))

    elif set_type == 'point':
        Z = Zonotope(cZ.c, cZ.G)
        I = Interval(Z)
        if hasattr(I, 'representsa_'):
            res = I.representsa_('point', tol)
        else:
            res = I.representsa_('point', tol)
        if res and kwargs.get('return_set', False):
            S = cZ.center()

    elif set_type == 'conPolyZono':
        # obviously true
        res = True

    elif set_type == 'conZonotope':
        # obviously true
        res = True
        if kwargs.get('return_set', False):
            S = cZ

    elif set_type == 'halfspace':
        # constrained zonotopes cannot be unbounded
        res = False

    elif set_type == 'interval':
        Z = Zonotope(cZ.c, cZ.G)
        if hasattr(Z, 'representsa_'):
            res = (cZ.A.size == 0) and Z.representsa_('interval', tol)
        else:
            res = (cZ.A.size == 0) and Z.representsa_('interval', tol)
        if res and kwargs.get('return_set', False):
            S = Interval(Z)

    elif set_type == 'polytope':
        res = True

    elif set_type == 'polyZonotope':
        res = True

    elif set_type == 'probZonotope':
        res = False

    elif set_type == 'zonoBundle':
        res = True

    elif set_type == 'zonotope':
        res = (cZ.A.size == 0)
        # note: there may be cases with constraints that can still be
        # represented by zonotopes
        if res and kwargs.get('return_set', False):
            S = Zonotope(cZ.c, cZ.G)

    elif set_type == 'hyperplane':
        # constrained zonotopes cannot be unbounded (unless 1D, where
        # hyperplane is also bounded)
        res = n == 1

    elif set_type == 'convexSet':
        res = True

    elif set_type == 'emptySet':
        res = _aux_isEmptySet(cZ, tol)
        if res and kwargs.get('return_set', False):
            from cora_python.contSet.emptySet import EmptySet
            S = EmptySet(n)

    elif set_type == 'fullspace':
        # constrained zonotopes cannot be unbounded
        res = False

    # unsupported types
    elif set_type in ['conHyperplane', 'capsule', 'ellipsoid', 'levelSet', 'parallelotope']:
        raise CORAerror('CORA:notSupported',
                       f'Comparison of conZonotope to {set_type} not supported.')
    else:
        res = False

    return _return_result(res, S)


def _aux_isEmptySet(cZ: 'ConZonotope', tol: float) -> bool:
    """Check if constrained zonotope represents empty set"""
    import numpy as np
    from cora_python.contSet.zonotope import Zonotope
    # If enclosing zonotope is empty, constrained zonotope is empty
    Z = Zonotope(cZ.c, cZ.G)
    if Z.representsa_('emptySet', tol):
        return True

    # No constraints -> not empty
    if cZ.A.size == 0:
        return False

    # Feasibility of A beta = b with beta in [-1,1]
    try:
        from scipy.optimize import linprog
        nrGen = cZ.G.shape[1]
        Aeq = cZ.A
        beq = cZ.b.flatten() if cZ.b.ndim > 1 else cZ.b
        bounds = [(-1.0, 1.0)] * nrGen
        # Minimize 0 subject to constraints
        res = linprog(np.zeros(nrGen), A_ub=None, b_ub=None, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
        return not bool(res.success)
    except Exception:
        # Fallback: use pseudoinverse feasibility check
        try:
            x0 = np.linalg.pinv(cZ.A) @ cZ.b
            return not np.all(np.abs(x0) <= 1 + tol)
        except Exception:
            return False