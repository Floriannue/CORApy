import numpy as np
from scipy.special import gamma
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

"""
volume_ - computes the volume of an ellipsoid acc. to Sec. 2 in [1]

Syntax:
   vol = volume_(E)

Inputs:
   E - ellipsoid object

Outputs:
   vol - volume

Example:
   E = ellipsoid([1,0;0,3],[1;-1]);
   vol = volume(E);

References:
   [1] A. Moshtagh. "Minimum volume enclosing ellipsoid", 2005

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/volume

Authors:       Victor Gassmann
Written:       28-August-2019
Last update:   04-July-2022 (VG, allow class array input)
               18-August-2022 (MW, include standardized preprocessing)
               05-October-2024 (MW, remove class array)
Last revision: 27-March-2023 (MW, rename volume_)
               Automatic python translation: Florian Nüssel BA 2025
"""

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