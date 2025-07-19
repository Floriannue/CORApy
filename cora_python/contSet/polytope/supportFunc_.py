"""
supportFunc_ - Calculate the upper or lower bound of a polytope object
   along a certain direction

Syntax:
   val = supportFunc_(P,dir,type)
   [val,x] = supportFunc_(P,dir,type)

Inputs:
   P - polytope object
   dir - direction for which the bounds are calculated (vector of size
         (n,1) )
   type - upper or lower bound ('lower', 'upper', 'range')

Outputs:
   val - bound of the polytope in the specified direction
   x - support vector

Example:
   A = [1 0; -1 1; -1 -1]; b = [1;1;1];
   P = polytope(A,b);
   [val,x] = supportFunc(P,[0;-1],'upper');

Reference:
   [1] M. Wetzlinger, V. Kotsev, A. Kulmburg, M. Althoff. "Implementation
       of Polyhedral Operations in CORA 2024", ARCH'24.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/supportFunc, conZonotope/supportFunc_

Authors:       Niklas Kochdumper, Victor Gassmann, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       19-November-2019 (MATLAB)
Last update:   16-March-2021 (added unbounded support) (MATLAB)
               10-December-2022 (MW, add 'range') (MATLAB)
               13-December-2022 (MW, add call to MOSEK) (MATLAB)
               15-November-2023 (MW, computation for vertex representation) (MATLAB)
               09-July-2024 (TL, added fallback during linprog) (MATLAB)
Python translation: 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope


def supportFunc_(P: 'Polytope', 
                dir: np.ndarray, 
                type_: str = 'upper',
                method: str = 'interval',
                max_order_or_splits: int = 8,
                tol: float = 1e-3) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Calculate the upper or lower bound of a polytope object along a certain direction
    
    Args:
        P: polytope object
        dir: direction for which the bounds are calculated (vector of size (n,1))
        type_: upper or lower bound ('lower', 'upper', 'range')
        method: method for computation (not used in this implementation)
        max_order_or_splits: maximum order or number of splits (not used)
        tol: tolerance (not used)
        
    Returns:
        Union[float, Tuple]: bound value, or tuple of (val, x) if requested
        
    Raises:
        ValueError: If invalid type
    """
    
    # Validate type
    if type_ not in ['lower', 'upper', 'range']:
        raise ValueError(f"Invalid type '{type_}'. Use 'lower', 'upper', or 'range'.")
    
    # Check if polytope represents fullspace (R^n)
    if P.representsa_('fullspace', 0):
        return _aux_supportFunc_fullspace(dir, type_)
    
    # For now, implement only the fullspace case
    # TODO: Implement vertex representation and linear programming cases
    raise NotImplementedError("supportFunc_ for polytope is only implemented for fullspace case")


def _aux_supportFunc_fullspace(dir: np.ndarray, type_: str) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Support function of a polytope that represents R^n
    
    Args:
        dir: direction vector
        type_: type of bound ('lower', 'upper', 'range')
        
    Returns:
        Union[float, Tuple]: bound value, or tuple of (val, x)
    """
    
    # Support vector: Inf * sign(dir), with NaN values set to 0
    x = np.inf * np.sign(dir)
    x[np.isnan(x)] = 0
    
    if type_ == 'upper':
        val = np.inf
    elif type_ == 'lower':
        val = -np.inf
    elif type_ == 'range':
        # Import here to avoid circular imports
        from cora_python.contSet.interval import Interval
        val = Interval(-np.inf, np.inf)
    
    return val, x

# ------------------------------ END OF CODE ------------------------------ 