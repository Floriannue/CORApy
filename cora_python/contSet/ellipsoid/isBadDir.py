"""
isBadDir - checks if specified directions are bad directions for
    the Minkowski difference of E1 and E2

Syntax:
   res = isBadDir(E1,E2,L)

Inputs:
   E1 - ellipsoid object
   E2 - ellipsoid object
   L  - (n x N) matrix of directions, where n is the set dimension, and N
         is the number of directions to check

Outputs:
   res - true/false for each direction

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Victor Gassmann
Written:       13-March-2019
Last update:   10-June-2022
               04-July-2022
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.helper.sets.contSet.ellipsoid.simdiag import simdiag
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid  # For dim method and type hinting
from typing import Union, List
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def isBadDir(E1: Ellipsoid, E2: Ellipsoid, L: np.ndarray) -> Union[bool, np.ndarray]:

    # input check
    # The inputArgsCheck in Python likely doesn't have the same @(L) syntax for custom checks
    # We'll rely on type hints and simple if statements for now.
    # The original MATLAB code does: {@(L) size(L,1) == dim(E1)}
    inputArgsCheck([
        [E1, 'att', 'ellipsoid'], # Removed 'scalar' attribute
        [E2, 'att', 'ellipsoid'], # Removed 'scalar' attribute
        # For L, we need to check if it's a numeric matrix and its first dimension matches dim(E1)
        [L, 'att', 'numeric', ['matrix']]
    ])

    # custom check for L's dimension
    if L.shape[0] != E1.dim():
        raise CORAerror("CORA:dimensionMismatch", f"L's first dimension ({L.shape[0]}) must match E1's dimension ({E1.dim()}).")

    # check dimension
    equal_dim_check(E1, E2)

    TOL = min(E1.TOL, E2.TOL)
    
    # simdiag returns T, D. We only need D.
    _, D = simdiag(E1.Q, E2.Q, TOL)
    
    # Ensure D is a 2D array even if it's a scalar from simdiag
    if D.ndim == 0:
        r = 1 / np.max(D)
    else:
        # MATLAB diag(D) extracts diagonal elements. np.diag does the same.
        # max(diag(D)) is simply np.max(np.diag(D))
        r = 1 / np.max(np.diag(D))

    res = np.full((1, L.shape[1]), False, dtype=bool)
    for i in range(L.shape[1]):
        l = L[:, i:i+1] # Slice to maintain column vector shape
        # MATLAB: sqrt(l'*E1.Q*l)/sqrt(l'*E2.Q*l) > r+TOL;
        # Python: np.sqrt(l.T @ E1.Q @ l) / np.sqrt(l.T @ E2.Q @ l) > r + TOL
        term1 = np.sqrt(l.T @ E1.Q @ l)
        term2 = np.sqrt(l.T @ E2.Q @ l)

        # Handle division by zero or very small numbers by adding a small epsilon
        # This prevents RuntimeWarning: invalid value encountered in true_divide
        epsilon = np.finfo(float).eps
        denominator = term2 if np.abs(term2) > epsilon else epsilon
        
        res[0, i] = (term1 / denominator) > (r + TOL)

    # If it's a single direction, return a single boolean, otherwise return the array
    if L.shape[1] == 1:
        return res[0, 0]
    else:
        return res 