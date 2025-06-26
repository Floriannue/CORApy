import numpy as np
from scipy.special import gamma
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

def volume_(E: Ellipsoid) -> float:
    """
    volume_ - computes the volume of an ellipsoid

    Syntax:
        vol = volume_(E)

    Inputs:
        E - ellipsoid object

    Outputs:
        vol - volume
    """

    if E.isemptyobject():
        return 0.0
    
    n = E.dim()

    # Volume of an n-dimensional unit ball
    v_ball = (np.pi**(n/2)) / gamma(n/2 + 1)

    # Volume of ellipsoid is V_ball * sqrt(det(Q))
    vol = v_ball * np.sqrt(np.linalg.det(E.Q))

    return vol 