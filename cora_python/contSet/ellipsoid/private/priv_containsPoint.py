"""
Private function for checking if an ellipsoid contains a point cloud.
"""

import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_containsPoint(E: 'Ellipsoid', S: np.ndarray, tol: float) -> Tuple[Union[bool, np.ndarray], Union[bool, np.ndarray], Union[float, np.ndarray]]:
    """
    Checks whether an ellipsoid contains a point cloud.
    
    Args:
        E: ellipsoid object
        S: point cloud (n x N array)
        tol: tolerance
        
    Returns:
        res: true/false for each point
        cert: certificate (always True for point containment)
        scaling: scaling factor for each point
    """
    
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    N = S.shape[1]
    
    # Handle empty point cloud
    if N == 0:
        # Return empty arrays (not scalars) to match MATLAB and avoid unpacking errors
        return np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=float)
    
    c = E.center()
    
    res = np.zeros(N, dtype=bool)
    cert = np.ones(N, dtype=bool)
    scaling = np.zeros(N)
    
    for i in range(N):
        scaling[i] = E.ellipsoidNorm(S[:, i:i+1] - c)
        print(f"Point {i} scaling: {scaling[i]}, Expected <= {1 + tol}") # Re-added print statement
        if scaling[i] <= 1 + tol:
            res[i] = True
        elif np.isnan(scaling[i]):
            # This can only happen when Q=0 and p=0, in which case we need
            # to manually set scaling and res
            res[i] = True
            scaling[i] = 0.0
    
    # Always return arrays, even for a single point, to match MATLAB behavior and avoid unpacking errors in calling code.
    # (MATLAB always returns arrays of length N, even for N=1.)
    return res, cert, scaling 