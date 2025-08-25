"""
calcSquaredGInd - computes the indices of multiplicative G1' * G2

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Tuple, Optional


def calcSquaredGInd(G1: np.ndarray, G2: np.ndarray, isEqual: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the indices of multiplicative G1' * G2.
    
    Args:
        G1: row vector of generator matrix 1
        G2: row vector of generator matrix 2
        isEqual: whether they are equal for optimizations (default: False)
        
    Returns:
        G1_ind: indices of G1 in the final matrix
        G2_ind: indices of G2 in the final matrix
        G1_ind2: indices of G1 in the final matrix which occur twice
        G2_ind2: indices of G2 in the final matrix which occur twice
        
    See also: nnHelper/calcSquaredG, nnHelper/calcSquaredE
    """
    if isEqual is None:
        isEqual = False
    
    # Create index matrices for G1 and G2
    tempG1 = np.power(np.arange(len(G1)).reshape(-1, 1), np.ones(len(G2)))  # 0-based indexing like Python
    tempG2 = np.power(np.arange(len(G2)), np.ones(len(G1)).reshape(-1, 1))  # 0-based indexing like Python
    
    if isEqual:
        # we can ignore the left lower triangle in this case
        # as it's the same as the right upper triangle
        # -> double right upper triangle
        
        # Create index vectors for G1 and G2
        G1_ind = np.arange(len(G1))  # 0-based indexing like Python
        G2_ind = np.arange(len(G2))  # 0-based indexing like Python
        
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(len(G1), k=1)
        G1_ind2 = tempG1[triu_indices].flatten()
        G2_ind2 = tempG2[triu_indices].flatten()
        
        # Remove zeros
        G1_ind2 = G1_ind2[G1_ind2 > 0]
        G2_ind2 = G2_ind2[G2_ind2 > 0]
        
    else:
        # calculate all values
        G1_ind = tempG1
        G2_ind = tempG2
        
        G1_ind = G1_ind.reshape(1, -1)
        G2_ind = G2_ind.reshape(1, -1)
    
    return G1_ind, G2_ind, G1_ind2, G2_ind2
