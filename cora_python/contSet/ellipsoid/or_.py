"""
or_ - overloads '|' operator to compute an inner/outer approximation of the union of an ellipsoid and another set representation

Syntax:
    S_out = E | S
    S_out = or_(E, S)
    S_out = or_(E, S, mode)

Inputs:
    E - ellipsoid object
    S - contSet object, numeric, or cell-array
    mode - type of approximation (optional):
               'inner' (inner-approximation)
               'outer' (outer-approximation)

Outputs:
    S_out - ellipsoid object

Example:
    E1 = Ellipsoid(np.array([[3, -1], [-1, 1]]), np.array([[1], [0]]))
    E2 = Ellipsoid(np.array([[5, 1], [1, 2]]), np.array([[1], [-1]]))
    E = E1 | E2

Authors:       Victor Gassmann (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       09-March-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.ellipsoid.private.priv_orEllipsoidOA import priv_orEllipsoidOA
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric as reorderNumeric
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def or_(E, S, mode=None):
    """
    Overloads '|' operator for ellipsoid union.
    Args:
        E: ellipsoid object
        S: contSet object, numeric, or list
        mode: 'inner' or 'outer' (default: 'outer')
    Returns:
        S_out: ellipsoid object
    """
    if mode is None:
        mode = 'outer'
    mode = setDefaultValues(['outer'], [mode])[0]

    # Ensure that numeric is second input argument
    E, S = reorderNumeric(E, S)

    # Check dimensions
    equal_dim_check(E, S)

    # Check input arguments
    inputArgsCheck([
        [E, 'att', 'ellipsoid'],
        [S, 'att', ['contSet', 'numeric', 'list']],
        [mode, 'str', ['outer', 'inner']]
    ])

    # Call function with lower precedence
    if hasattr(S, 'precedence') and S.precedence < E.precedence:
        return S.or_(E, mode)

    # Handle special cases for empty sets first, as they don't involve SDP.
    # This part should ideally be in or.py, but temporarily here for debugging priv_orEllipsoidOA.
    # If E is empty and S is not, return S
    if E.isemptyobject():
        return S
    # If S is empty and E is not, return E
    elif S.isemptyobject():
        return E

    # Now, if both are non-empty, proceed with the union via SDP or raise NotImplementedError
    # For the `or_` function, we expect a robust SDP solver. Since open-source
    # solvers struggle with the underlying LMI for general ellipsoids (even N=2),
    # we will raise a NotImplementedError here for now.
    # The actual SDP logic is in priv_orEllipsoidOA.py, which will also raise
    # CORAerror on solver failure.
    # raise NotImplementedError(
    #     'Computing the outer-approximation of the union of non-empty ellipsoids ' +
    #     'requires a robust SDP solver (e.g., commercial solvers like MOSEK) or ' +
    #     'a more advanced, numerically stable open-source formulation. ' +
    #     'This functionality is currently not fully supported with standard ' +
    #     'open-source CVXPY solvers.'
    # )

    # This code block below is commented out because the functionality is being
    # disabled due to solver limitations, as described in the NotImplementedError above.
    # The logic is preserved in priv_orEllipsoidOA.py for reference and future work.
    
    # All to one list
    if not isinstance(S, list):
        S = [S]
    
    # Verify contents of S (ellipsoid, polytope, or numeric column vector)
    from cora_python.contSet.contSet.contSet import ContSet # Import here to avoid circular dependency
    from cora_python.contSet.polytope.polytope import Polytope # Import here to avoid circular dependency
    
    for s_i in S:
        if not (isinstance(s_i, (ContSet, np.ndarray))):
            if not (isinstance(s_i, np.ndarray) and s_i.ndim == 2 and s_i.shape[1] == 1):
                raise CORAerror('CORA:noops', E, S, mode)
        elif isinstance(s_i, ContSet):
            if not (isinstance(s_i, Ellipsoid) or isinstance(s_i, Polytope)):
                raise CORAerror('CORA:noops', E, S, mode)


    E_cell = [E] + [aux_convert(s_i) for s_i in S]
    return priv_orEllipsoidOA(E_cell)


def aux_convert(S):
    # Helper function to convert multiple operands to ellipsoids correctly
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
    from cora_python.contSet.polytope.polytope import Polytope # Moved here
    if isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == 1:
        return Ellipsoid.origin(len(S)) + S
    elif isinstance(S, Polytope):
        return Ellipsoid(S, 'outer')
    else:
        return S 