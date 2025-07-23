"""
priv_normalizeConstraints - normalizes the constraints

Syntax:
   [A,b,Ae,be] = priv_normalizeConstraints(A,b,Ae,be,type)

Inputs:
   A - inequality constraint matrix
   b - inequality constraint offset
   Ae - equality constraint matrix
   be - equality constraint offset
   type - (optional) 'b', 'be': normalize offset vectors b and be
                      'A', 'Ae': normalize norm of constraints in A and Ae to 1

Outputs:
   A - inequality constraint matrix
   b - inequality constraint offset
   Ae - equality constraint matrix
   be - equality constraint offset

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       03-October-2024 (MATLAB)
Last update:   ---
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Optional
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

def priv_normalizeConstraints(A: Optional[np.ndarray], b: Optional[np.ndarray], 
                              Ae: Optional[np.ndarray], be: Optional[np.ndarray], 
                              type_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    A_out = A.copy() if A is not None and A.size > 0 else np.array([]).reshape(0,0)
    b_out = b.copy() if b is not None and b.size > 0 else np.array([]).reshape(0,1)
    Ae_out = Ae.copy() if Ae is not None and Ae.size > 0 else np.array([]).reshape(0,0)
    be_out = be.copy() if be is not None and be.size > 0 else np.array([]).reshape(0,1)

    # Ensure b_out and be_out are 2D column vectors if not empty
    if b_out.ndim == 1:
        b_out = b_out.reshape(-1, 1)
    if be_out.ndim == 1:
        be_out = be_out.reshape(-1, 1)

    if type_str == 'b' or type_str == 'be':
        # normalize offset vectors b and be to -1|0|1

        # normalize inequality constraints
        if A_out.size > 0:
            # inequality constraints where A(:,i)*x <= 0 are left unchanged
        
            # find indices of constraints with b > 0 and b < 0
            idx_plus = b_out > 0
            idx_neg = b_out < 0
        
            # divide constraints with b > 0 by b
            if np.any(idx_plus):
                # Ensure broadcasting is handled correctly: divide row by row
                A_out[idx_plus.flatten(), :] = A_out[idx_plus.flatten(), :] / b_out[idx_plus.flatten()]
                b_out[idx_plus.flatten()] = 1
        
            # divide constraints with b < 0 by |b|
            if np.any(idx_neg):
                A_out[idx_neg.flatten(), :] = A_out[idx_neg.flatten(), :] / np.abs(b_out[idx_neg.flatten()])
                b_out[idx_neg.flatten()] = -1
        
    
        # normalize equality constraints
        if Ae_out.size > 0:
            # equality constraints where Ae(:,i)*x = 0 are left unchanged
        
            # find indices of constraints with be > 0 or be < 0
            idx_nonzero = (be_out > 0) | (be_out < 0)
        
            # divide constraints with be =!= 0 by be
            if np.any(idx_nonzero):
                Ae_out[idx_nonzero.flatten(), :] = Ae_out[idx_nonzero.flatten(), :] / be_out[idx_nonzero.flatten()]
                be_out[idx_nonzero.flatten()] = 1
            
    elif type_str == 'A' or type_str == 'Ae':
        # normalize norm of constraints in A and Ae to 1
        # skip constraints of the form 0*x <= ... or 0*x == ...

        # normalize inequality constraints
        if A_out.size > 0:
            # vecnorm(X', 2, 1) in MATLAB is norm of each column of X', which is row norm of X
            normA = np.linalg.norm(A_out, axis=1) # Compute row-wise 2-norm
            idx_nonzero = ~withinTol(normA, 0) # Boolean mask for non-zero norms
            
            if np.any(idx_nonzero):
                # Normalize rows by dividing by their norms
                # Use np.newaxis to enable broadcasting division of (N,) array by (N,1) column vector
                A_out[idx_nonzero, :] = A_out[idx_nonzero, :] / normA[idx_nonzero, np.newaxis]
                b_out[idx_nonzero] = b_out[idx_nonzero] / normA[idx_nonzero, np.newaxis]
        
        # normalize equality constraints
        if Ae_out.size > 0:    
            normA = np.linalg.norm(Ae_out, axis=1)
            idx_nonzero = ~withinTol(normA, 0)
            if np.any(idx_nonzero):
                Ae_out[idx_nonzero, :] = Ae_out[idx_nonzero, :] / normA[idx_nonzero, np.newaxis]
                be_out[idx_nonzero] = be_out[idx_nonzero] / normA[idx_nonzero, np.newaxis]

    return A_out, b_out, Ae_out, be_out 