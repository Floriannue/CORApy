import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

def transform(E: 'Ellipsoid', M: np.ndarray) -> 'Ellipsoid':
    """
    transform - performs an affine transformation on an ellipsoid.

    Syntax:
        E = transform(E,M)

    Inputs:
        E - ellipsoid object
        M - transformation matrix

    Outputs:
        E - transformed ellipsoid

    Examples:
        E = Ellipsoid(np.eye(2));
        E = transform(E,np.array([[2,0],[0,1]]));

    Authors:       Matthias Althoff
    Written:       13-March-2019
    Last update:   16-March-2021 (VT, considering an array of ellipsoids)
    Last revision: ---
    Automatic python translation: Florian Nüssel BA 2025
    """

    # new shape matrix
    Q_new = M @ E.Q @ M.T
    
    # new center
    q_new = M @ E.q

    # construct new ellipsoid
    E_new = E.__class__(Q_new, q_new, E.TOL)
    
    return E_new