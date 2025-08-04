import numpy as np
from scipy.optimize import root_scalar
from .priv_rootfnc import priv_rootfnc # Import the actual priv_rootfnc

def priv_compIntersectionParam(W1: np.ndarray, q1: np.ndarray, W2: np.ndarray, q2: np.ndarray) -> float:
    """
    priv_compIntersectionParam - computes zero root of 'priv_rootfnc' and
    returns the corresponding argument

    Syntax:
        val = priv_compIntersectionParam(W1,q1,W2,q2)

    Inputs:
        W1,q,W2,q2 - center and shape matrices of E1,E2 (see 'and')

    Outputs:
        val - solution minimizing the volume of the parametrization in 'and'
              (for details, see [1],[2])

    References:
      [1] Largely based on the Ellipsoidal Toolbox, see:
          https://de.mathworks.com/matlabcentral/fileexchange/21936-ellipsoidal-toolbox-et
      [2] For detailed explanations, see:
          https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-46.pdf

    Authors:       Victor Gassmann
    Written:       14-October-2019
    Last update:   20-May-2022 (VG, use interval functionality of fzero)
    Last revision: ---
                   Automatic python translation: Florian Nüssel BA 2025
    """

    f = lambda p: priv_rootfnc(p, W1, q1, W2, q2)

    try:
        # Use bracket method to find root within [0,1]
        sol = root_scalar(f, bracket=[0, 1], method='brentq')
        val = sol.root
    except ValueError: # Occurs if f(0) and f(1) have the same sign (no root in bracket)
        # choose the one extremum which has smallest volume
        # In MATLAB, det(W) is used, which is inverse of volume. So larger det(W) means smaller volume.
        if np.linalg.det(W1) > np.linalg.det(W2):
            val = 1.0
        else:
            val = 0.0
    except Exception as e:
        # Catch other potential errors from root_scalar
        print(f"Warning: root_scalar failed with: {e}. Falling back to volume comparison.")
        if np.linalg.det(W1) > np.linalg.det(W2):
            val = 1.0
        else:
            val = 0.0

    return val