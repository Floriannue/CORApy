"""
priv_compact_alignedIneq - removes all redundant aligned constraints

Description:
    Removes all redundant aligned inequality constraints by keeping only
    the one with the smallest offset for each group of aligned constraints.

Syntax:
    A, b = priv_compact_alignedIneq(A, b, tol)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    tol - tolerance

Outputs:
    A - inequality constraint matrix
    b - inequality constraint offset

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.auxiliary.withinTol import withinTol


def priv_compact_alignedIneq(A, b, tol):
    """
    Removes all redundant aligned constraints
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        tol: tolerance
        
    Returns:
        A: inequality constraint matrix
        b: inequality constraint offset
    """
    if A is None or A.size == 0:
        return A, b
    
    # Sort the matrix rows to detect aligned normal vectors
    sorted_indices = np.lexsort(A.T)
    A = A[sorted_indices]
    b = b[sorted_indices]
    
    # Remove all aligned halfspaces
    counter = 0
    cTemp = 0
    A_ = np.zeros_like(A)
    b_ = np.zeros_like(b)
    
    while counter < A.shape[0] - 1:
        # Check if two normal vectors are identical
        if np.sum(np.abs(A[counter, :] - A[counter + 1, :])) < tol:
            a_ = A[counter, :]
            
            # Determine minimum b
            bmin = min(b[counter], b[counter + 1])
            counter += 2
            
            while counter < A.shape[0]:
                if np.sum(np.abs(A[counter, :] - a_)) < tol:
                    bmin = min(bmin, b[counter])
                    counter += 1
                else:
                    break
            
            # Store unified normal vector
            A_[cTemp, :] = a_
            b_[cTemp] = bmin
            cTemp += 1
        else:
            A_[cTemp, :] = A[counter, :]
            b_[cTemp] = b[counter]
            cTemp += 1
            counter += 1
    
    # Add last element
    if counter == A.shape[0] - 1:
        A_[cTemp, :] = A[-1, :]
        b_[cTemp] = b[-1]
        cTemp += 1
    
    # Override constraint set
    A = A_[:cTemp, :]
    b = b_[:cTemp]
    
    return A, b 