"""
This module contains the function for getting the generator matrix of an ellipsoid.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def generators(E: 'Ellipsoid') -> np.ndarray:
    """
    Returns the generator matrix of an ellipsoid in generator representation.
    
    Args:
        E: ellipsoid object
        
    Returns:
        G: generator matrix
    """
    from cora_python.contSet.ellipsoid.dim import dim
    
    Q = E.Q
    if Q is None or Q.size == 0:
        return np.zeros((dim(E), 0))
    
    U, D, _ = np.linalg.svd(Q)
    G = U @ np.sqrt(np.diag(D))
    
    return G 