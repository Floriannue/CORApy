"""
isemptyobject - checks whether an ellipsoid contains any information at
    all; consequently, the set is interpreted as the empty set 

Syntax:
    res = isemptyobject(E)

Inputs:
    E - ellipsoid object

Outputs:
    res - true/false

Example: 
    E = Ellipsoid([[1,0],[0,2]], [0,1])
    isemptyobject(E)  # False

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       24-July-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid


def isemptyobject(E: 'Ellipsoid') -> bool:
    """
    Checks whether an ellipsoid contains any information at all
    
    Args:
        E: ellipsoid object
        
    Returns:
        res: True if ellipsoid is empty, False otherwise
    """
    return _aux_check_if_empty(E)


def _aux_check_if_empty(E: 'Ellipsoid') -> bool:
    """
    Auxiliary function to check if ellipsoid is empty
    
    Args:
        E: ellipsoid object
        
    Returns:
        res: True if ellipsoid is empty, False otherwise
    """
    # Check if Q is empty (0,0) - this indicates an empty ellipsoid
    # For empty ellipsoids: Q is (0,0) and q is (n,0) where n >= 0
    # Both have size == 0, but Q.shape == (0,0) while q.shape == (n,0)
    return (hasattr(E, 'Q') and isinstance(E.Q, np.ndarray) and 
            E.Q.shape == (0, 0) and
            hasattr(E, 'q') and isinstance(E.q, np.ndarray) and 
            E.q.size == 0 and E.q.ndim == 2 and E.q.shape[1] == 0) 