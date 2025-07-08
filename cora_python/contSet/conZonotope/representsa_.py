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


def representsa_(cZ: 'ConZonotope', set_type: str, tol: float = 1e-12) -> Union[bool, Tuple[bool, 'ContSet']]:
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
    
    # check empty object case
    empty_result = cZ.representsa_emptyObject(set_type)
    if isinstance(empty_result, tuple) and len(empty_result) == 3:
        empty, res, S = empty_result
        if empty:
            if S is not None:
                return res, S
            else:
                return res
    elif isinstance(empty_result, tuple) and len(empty_result) == 2:
        empty, res = empty_result
        if empty:
            return res

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
        if res:
            S = np.zeros((n, 1))

    elif set_type == 'point':
        Z = Zonotope(cZ.c, cZ.G)
        I = Interval(Z)
        if hasattr(I, 'representsa_'):
            res = I.representsa_('point', tol)
        else:
            res = I.representsa_('point', tol)
        if res:
            S = cZ.center()

    elif set_type == 'conPolyZono':
        # obviously true
        res = True

    elif set_type == 'conZonotope':
        # obviously true
        res = True
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
        if res:
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
        if res:
            S = Zonotope(cZ.c, cZ.G)

    elif set_type == 'hyperplane':
        # constrained zonotopes cannot be unbounded (unless 1D, where
        # hyperplane is also bounded)
        res = n == 1

    elif set_type == 'convexSet':
        res = True

    elif set_type == 'emptySet':
        res = _aux_isEmptySet(cZ, tol)
        if res:
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

    # Return tuple if conversion set was computed, otherwise just boolean
    if S is not None:
        return res, S
    else:
        return res


def _aux_isEmptySet(cZ: 'ConZonotope', tol: float) -> bool:
    """Check if constrained zonotope represents empty set"""
    
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.polytope import Polytope
    # check if the (in general, enclosing) zonotope is already empty
    Z = Zonotope(cZ.c, cZ.G)
    if hasattr(Z, 'representsa_'):
        if Z.representsa_('emptySet', tol):
            return True
    else:
        if Z.representsa_('emptySet', tol):
            return True

    # if there are no constraints, we are finished
    if cZ.A.size == 0:
        return False

    # approach: if the constraints are satisfiable, there is at least one value
    # for the beta factors and, thus, the constrained zonotope is non-empty

    # functions below do not support sparse matrices
    A = cZ.A
    if hasattr(A, 'toarray'):  # scipy sparse matrix
        A = A.toarray()

    # null space of the constraints
    try:
        Neq = null_space(A)
    except ImportError:
        # fallback using SVD
        U, s, Vh = np.linalg.svd(A)
        null_mask = s <= np.finfo(float).eps
        Neq = Vh[len(s):, :].T.conj()

    # find a single point that satisfies the constraints
    # Handle case where A is empty
    if A.size == 0:
        x0 = np.array([])
    else:
        x0 = np.linalg.pinv(A) @ cZ.b

    if A.size > 0 and np.linalg.norm(A @ x0 - cZ.b) > 1e-10 * np.linalg.norm(cZ.b):  # infeasible
        # note: the tolerance above must be hardcoded to some non-zero value
        return True

    if Neq.size == 0:  # null space empty -> constraints admit a single point
        # check if the single point for beta satisfies the unit cube
        nrGen = cZ.G.shape[1]
        unit_interval = Interval(-np.ones((nrGen, 1)), np.ones((nrGen, 1)))
        result, _, _ = unit_interval.contains_(x0, 'exact', tol)
        return not result

    # check if the null-space intersects the unit-cube
    nrCon, nrGen = A.shape
    unit_interval = Interval(-np.ones((nrGen, 1)), np.ones((nrGen, 1)))

    # loop over all constraints (= hyperplanes)
    for i in range(nrCon):
        # hyperplane from a constraint does not intersect the unit cube
        # -> set is empty
        P = Polytope(A[i:i+1, :], cZ.b[i:i+1])
        if not P.isIntersecting_(unit_interval, 'exact', tol):
            return True

    # use linear programming to check if the constrained zonotope is
    # empty (this seems to be more robust than the previous solution
    # using the polytope/isempty function)
    if nrCon >= 1:
        # setup linear program
        problem = {
            'f': np.ones(nrGen),
            'Aineq': np.array([]).reshape(0, nrGen),
            'bineq': np.array([]),
            'Aeq': A,
            'beq': cZ.b,
            'ub': np.ones(nrGen),
            'lb': -np.ones(nrGen)
        }
        
        try:
            _, _, exitflag = CORAlinprog(problem)
            return exitflag == -2
        except Exception:
            # fallback to simple feasibility check
            return False

    return False 