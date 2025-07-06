"""
plus - Overloaded '+' operator for approximating the Minkowski sum of an
    ellipsoid and another set or point

Syntax:
    S_out = E + S
    S_out = plus(E,S)
    S_out = plus(E,S,mode)
    S_out = plus(E,S,mode,L)

Inputs:
    E - ellipsoid object, numeric
    S - contSet object (or cell array), numeric
    mode - (optional) type of approximation
               'inner'
               'outer':
               'outer:halder': available when L is empty
    L - (optional) directions to use for approximation

Outputs:
    S_out - set after Minkowski sum

Example: 
    E1 = ellipsoid(eye(2),[1;-1])
    E2 = ellipsoid(diag([1,2]))
    Ep = E1 + E2
    figure hold on
    plot(E1) plot(E2)
    plot(Ep,[1,2],'r')

References:
   [1] Kurzhanskiy, A.A. and Varaiya, P., 2006, December. Ellipsoidal
       toolbox (ET). In Proceedings of the 45th IEEE Conference on
       Decision and Control (pp. 1498-1503). IEEE.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: priv_plusEllipsoid

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       09-March-2021 (MATLAB)
Last update:   04-July-2022 (VG, class array instead of cell array, MATLAB)
               17-March-2023 (MW, simplify argument pre-processing, MATLAB)
               05-October-2024 (MW, remove class array, MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric as reorderNumeric
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck


def plus(factor1, factor2, *args):
    """
    Overloaded '+' operator for approximating the Minkowski sum of an
    ellipsoid and another set or point
    
    Args:
        factor1: ellipsoid object or numeric
        factor2: contSet object, cell array, or numeric  
        mode: type of approximation ('inner', 'outer', 'outer:halder')
        L: directions to use for approximation
        
    Returns:
        S_out: set after Minkowski sum
    """
    # Ensure that numeric is second input argument
    E, S = reorderNumeric(factor1, factor2)
    
    # Default values
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Ellipsoid':
        defaults, _ = setDefaultValues(['outer:halder', np.zeros((E.dim(), 0))], args)
        mode, L = defaults
    else:
        defaults, _ = setDefaultValues(['outer', np.zeros((E.dim(), 0))], args)
        mode, L = defaults
    
    # Check input arguments
    inputArgsCheck([
        [E, 'att', 'ellipsoid'],
        [S, 'att', ['cell', 'contSet', 'numeric']],
        [mode, 'str', ['outer', 'outer:halder', 'inner']],
        [L, 'att', 'numeric']
    ])
    
    # Minkowski addition with empty set
    if E.representsa_('emptySet', E.TOL):
        return Ellipsoid.empty(E.dim())
    
    if not isinstance(S, list) and hasattr(S, 'representsa_') and S.representsa_('emptySet', E.TOL):
        return Ellipsoid.empty(E.dim())
    
    if not isinstance(S, list) and hasattr(S, 'representsa_') and S.representsa_('origin', E.TOL):
        # Adding the origin does not change the set
        return E.copy()
    
    # Dimension checks
    equalDimCheck(E, S)
    equalDimCheck(E, L)
    
    # Addition of vector
    if isinstance(S, np.ndarray) and S.ndim <= 2 and (S.shape[1] == 1 if S.ndim == 2 else True):
        S_vec = S.reshape(-1, 1) if S.ndim == 1 else S
        return Ellipsoid(E.Q, E.q + S_vec)
    
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Ellipsoid':
        # For now, return NotImplementedError until we translate private functions
        raise NotImplementedError("Ellipsoid-ellipsoid addition requires priv_plusEllipsoid which is not yet translated")
    
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Interval':
        # Convert to ellipsoid
        raise NotImplementedError("Interval addition requires ellipsoid(interval) conversion which is not yet translated")
    
    if hasattr(S, '__class__') and S.__class__.__name__ == 'Zonotope':
        # Convert to ellipsoid via interval enclosure
        raise NotImplementedError("Zonotope addition requires ellipsoid(interval(zonotope)) conversion which is not yet translated")
    
    if hasattr(S, '__class__') and S.__class__.__name__ == 'ConPolyZono':
        return S + E
    
    # Cell array case
    if isinstance(S, list):
        raise NotImplementedError("Cell array addition requires priv_plusEllipsoid which is not yet translated")
    
    # Throw error for all other combinations
    raise TypeError(f"Operation 'plus' not supported between instances of '{type(E).__name__}' and '{type(S).__name__}'") 