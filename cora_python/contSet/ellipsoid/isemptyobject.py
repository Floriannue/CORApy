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
   E = Ellipsoid(np.array([[1,0],[0,2]]),np.array([[0],[1]]));
   E.isemptyobject(); # false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       24-July-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

def isemptyobject(self: Ellipsoid) -> bool:
    """
    Checks whether an ellipsoid contains any information at all.
    Consequently, the set is interpreted as the empty set.
    
    Args:
        self: The ellipsoid object.
        
    Returns:
        bool: True if the object is empty, False otherwise.
    """

    # Corresponding to MATLAB's aux_checkIfEmpty function logic:
    # isnumeric(E.Q) && isempty(E.Q)
    # && isnumeric(E.q) && isempty(E.q)
    # && isnumeric(E.TOL) && 
    # ( (isscalar(E.TOL) && E.TOL == 1e-6) || isempty(E.TOL) );

    # In MATLAB, this function handles class arrays (e.g., E(i,j)).
    # In Python, for a method, 'self' is a single instance. If the intent is
    # to apply this to an array of ellipsoids, we need to replicate that logic.
    # For now, we assume 'self' is a single Ellipsoid object.
    
    res_Q_empty = isinstance(self.Q, np.ndarray) and self.Q.size == 0
    res_q_empty = isinstance(self.q, np.ndarray) and self.q.shape[1] == 0 # Check for n x 0 matrix
    
    res_tol = False
    # MATLAB: (isscalar(E.TOL) && E.TOL == 1e-6) || isempty(E.TOL)
    if np.isscalar(self.TOL) and np.isclose(self.TOL, 1e-6):
        res_tol = True
    elif (isinstance(self.TOL, np.ndarray) and self.TOL.size == 0): # isempty(E.TOL) in MATLAB for ndarray
        res_tol = True

    # An ellipsoid is considered empty if its shape matrix Q is empty (0x0),
    # its center vector q is empty (0x0), and its tolerance is empty or default.
    return res_Q_empty and res_q_empty and res_tol