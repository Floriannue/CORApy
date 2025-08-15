"""
compareMatrices - checks if a given matrix has the same columns as
    another matrix (up to a given tolerance, sign, and potentially in
    different order) or whether its columns are a subset of the columns of
    the other matrix (same conditions); we assume no redundancies

Syntax:
    compareMatrices(M1,M2)
    compareMatrices(M1,M2,tol)
    compareMatrices(M1,M2,tol,flag)
    compareMatrices(M1,M2,tol,flag,ordered)
    compareMatrices(M1,M2,tol,flag,ordered,signed)

Inputs:
    M1 - matrix
    M2 - matrix
    tol - (optional) tolerance for numerical comparison
    flag - (optional) type of comparison
           'equal': M1 has to be exactly M2
           'subset': M1 only has to be a subset of M2
           default: 'equal'
    ordered - (optional) true/false, whether columns have to be in order
           default: false
    signed - (optional) true/false, whether columns are equal up to *-1

Outputs:
    - res - logical value

Example:
    M1 = [[2, 1], [0, 2]]
    M2 = [[1, 2], [2, 0]]
    M3 = [[2, 2], [0, 2]]
    compareMatrices(M1,M2) # true
    compareMatrices(M1,M3) # false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: display

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       13-November-2022 (MATLAB)
Last update:   22-November-2022 (MW, add subset variant, MATLAB)
               08-May-2023 (TL, ordered, MATLAB)
               19-January-2024 (MW, add signed, MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def compareMatrices(M1, M2, tol=None, flag='equal', ordered=False, signed=True):
    """
    Checks if a given matrix has the same columns as another matrix.
    
    Args:
        M1: First matrix
        M2: Second matrix
        tol: (optional) tolerance for numerical comparison (default: eps)
        flag: (optional) type of comparison ('equal' or 'subset', default: 'equal')
        ordered: (optional) whether columns have to be in order (default: False)
        signed: (optional) whether columns are equal up to *-1 (default: True)
        
    Returns:
        bool: True if matrices satisfy the comparison criteria
    """
    # Set default tolerance
    if tol is None:
        tol = np.finfo(float).eps
    
    # Handle None matrices
    if M1 is None and M2 is None:
        return True
    if M1 is None or M2 is None:
        return False
    
    # Initialize result
    res = True
    
    # Convert to numpy arrays and ensure they are 2D
    M1 = np.asarray(M1)
    if M1.ndim == 1:
        M1 = M1.reshape(-1, 1)
        
    M2 = np.asarray(M2)
    if M2.ndim == 1:
        M2 = M2.reshape(-1, 1)
    
    # Handle empty matrices
    if M1.size == 0 and M2.size == 0:
        return True
    
    # Check if matrices have same number of rows
    if M1.shape[0] != M2.shape[0]:
        return False
    elif flag == 'equal' and M1.shape[1] != M2.shape[1]:
        # Number of columns has to be equal
        return False
    elif flag == 'subset' and M1.shape[1] > M2.shape[1]:
        # Number of columns cannot be larger
        return False
    
    # Speed up computation if matrices have to be equal
    if ordered and signed and (flag == 'equal' or M1.shape == M2.shape):
        return withinTol(M1, M2, tol).all()
    
    # Logical indices which columns have been checked
    M2logical = np.zeros(M2.shape[1], dtype=bool)
    # Store index to start searching in M2; used for ordered=True
    # Has to be larger than index of last found column
    jmin = 0
    
    for i in range(M1.shape[1]):
        # Take i-th column of M1 and see if it is part of M2
        found = False
        for j in range(jmin, M2.shape[1]):
            if not M2logical[j]:
                # Check if columns match (with or without sign)
                if signed:
                    col_match = withinTol(M1[:, i], M2[:, j], tol).all() or withinTol(M1[:, i], -M2[:, j], tol).all()
                else:
                    col_match = withinTol(M1[:, i], M2[:, j], tol).all()
                if col_match:
                    found = True
                    M2logical[j] = True
                    if ordered:
                        jmin = j + 1
                    break
        
        # Exit if no corresponding column found
        if not found:
            return False
    
    if flag == 'equal':
        # All columns have to be found
        res = M2logical.all()
    elif flag == 'subset':
        # Not all columns have to be found
        res = True
    
    return res 