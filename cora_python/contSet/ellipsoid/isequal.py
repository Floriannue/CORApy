"""
isequal - checks if an ellipsoid is equal to another set or point

Syntax:
   res = isequal(E,S)
   res = isequal(E,S,tol)

Inputs:
   E - ellipsoid object
   S - contSet object, numeric
   tol - (optional) tolerance

Outputs:
   res - true/false

Example:
   E1 = Ellipsoid(np.array([[1,0],[0,1/2]]),np.array([[1],[1]]));
   E2 = Ellipsoid(np.array([[1+1e-15,0],[0,1/2]]),np.array([[1],[1]]));
   res = isequal(E1,E2);

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       13-March-2019 (MATLAB)
Last update:   04-July-2022 (VG, class array case, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Any, Tuple, Optional, List, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
    from cora_python.contSet.contSet.contSet import ContSet # For type hinting ContSet

def isequal(E: 'Ellipsoid', S: Union['ContSet', np.ndarray, float, int], tol: Optional[float] = None) -> bool:
    log.debug(f"isequal: E={E}, S={S}, tol={tol}")

    # Set default values for tol
    if tol is None:
        if hasattr(S, 'TOL'):
            tol = min(E.TOL, S.TOL)
        else: # S is numeric, use E's tolerance
            tol = E.TOL
    
    log.debug(f"isequal: After tol default, tol={tol}")

    # Check input arguments
    log.debug(f"isequal: Calling inputArgsCheck with E={E}, S={S}, tol={tol}")
    inputArgsCheck([
        [E, 'att', ['ellipsoid', 'numeric'], 'scalar'],
        [S, 'att', ['ContSet', 'numeric'], 'scalar'],
        [tol, 'att', 'numeric', ['nonnan', 'nonnegative', 'scalar']]
    ])
    log.debug(f"isequal: inputArgsCheck passed.")

    # ensure that numeric is second input argument
    E, S = reorder_numeric(E, S)
    log.debug(f"isequal: After reorder_numeric, E={E}, S={S}")
    
    # call function with lower precedence
    if hasattr(S, 'precedence') and isinstance(S, (E.__class__, S.__class__)) and S.precedence < E.precedence:
        log.debug(f"isequal: Delegating to S.isequal due to precedence. S={S}, E={E}")
        return S.isequal(E, tol)

    # ambient dimensions must match
    if not equal_dim_check(E, S, True):
        log.debug(f"isequal: Dimensions do not match. E.dim()={E.dim()}, S.dim()={getattr(S, 'dim', lambda: S.size if isinstance(S, np.ndarray) and S.ndim == 0 else S.shape[0])()}") # Handle numeric S for dim
        return False
    
    # ellipsoid-ellipsoid case
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid # avoid circular import
    if isinstance(S, Ellipsoid):
        log.debug(f"isequal: S is an Ellipsoid. Calling _aux_isequal_ellipsoid.")
        res = _aux_isequal_ellipsoid(E, S, tol)
        return res
    
    # If S is numeric (handled by reorder_numeric to be second arg)
    if isinstance(S, (int, float, np.ndarray)):
        log.debug(f"isequal: S is numeric. E.dim()={E.dim()}, S={S}")
        if E.dim() == 1 and isinstance(S, (int, float)): # 1D ellipsoid vs scalar
            log.debug(f"isequal: 1D Ellipsoid vs scalar numeric.")
            from cora_python.contSet.interval import Interval
            # Note: This Interval constructor call might not be ideal if S is truly a scalar point
            # and E is a general 1D ellipsoid. MATLAB's behavior is more complex here.
            # For now, rely on representsa_('point', tol) directly.
            return E.representsa_('point', tol) and np.allclose(E.q, np.array([[S]]), atol=tol)
        elif isinstance(S, np.ndarray) and S.ndim == 2 and (S.shape[0] == 1 or S.shape[1] == 1) and E.dim() == S.size:
             log.debug(f"isequal: Ellipsoid vs 2D row/column vector (point representation).")
             return E.representsa_('point', tol) and np.allclose(E.q, S.reshape(-1,1), atol=tol)
        else:
             log.debug(f"isequal: Generic numeric input not handled for point comparison. Raising CORA:noops.")
             raise CORAerror('CORA:noops', E, S)


    log.debug(f"isequal: S is an unhandled ContSet type. Raising CORA:noops.")
    raise CORAerror('CORA:noops', E, S)

def _aux_isequal_ellipsoid(E: 'Ellipsoid', S: 'Ellipsoid', tol: float) -> bool:
    log.debug(f"_aux_isequal_ellipsoid: E={E}, S={S}, tol={tol}")
    res = False

    if E.dim() != S.dim():
        log.debug(f"_aux_isequal_ellipsoid: Dimension mismatch. E.dim()={E.dim()}, S.dim()={S.dim()}")
        return res

    E_empty = E.representsa_('emptySet', np.finfo(float).eps)
    S_empty = S.representsa_('emptySet', np.finfo(float).eps)
    log.debug(f"_aux_isequal_ellipsoid: E_empty={E_empty}, S_empty={S_empty}")

    if (E_empty and not S_empty) or (not E_empty and S_empty):
        log.debug(f"_aux_isequal_ellipsoid: One empty, one not. Returning False.")
        return False
    elif E_empty and S_empty:
        log.debug(f"_aux_isequal_ellipsoid: Both empty. Returning True.")
        return True

    res = np.all(withinTol(E.Q, S.Q, tol)) and np.all(withinTol(E.q, S.q, tol))
    log.debug(f"_aux_isequal_ellipsoid: Numerical comparison result: {res}")
    
    return res 