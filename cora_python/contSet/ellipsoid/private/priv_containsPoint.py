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
    from cora_python.contSet.ellipsoid.ellipsoidNorm import ellipsoidNorm
    
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    N = S.shape[1]
    
    # Handle empty point cloud
    if N == 0:
        return True, True, 0.0
    
    c = E.center()
    
    res = np.zeros(N, dtype=bool)
    cert = np.ones(N, dtype=bool)
    scaling = np.zeros(N)
    
    for i in range(N):
        scaling[i] = ellipsoidNorm(E, S[:, i:i+1])
        if scaling[i] <= 1 + tol:
            res[i] = True
        elif np.isnan(scaling[i]):
            # This can only happen when Q=0 and p=0, in which case we need
            # to manually set scaling and res
            res[i] = True
            scaling[i] = 0.0
    
    if N == 1:
        return bool(res[0]), bool(cert[0]), float(scaling[0])
    return res, cert, scaling 