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
    
    # Initialize empty arrays
    G1_ind = np.array([])
    G2_ind = np.array([])
    G1_ind2 = np.array([])
    G2_ind2 = np.array([])
    
    if G1.size > 0 and G2.size > 0:
        # MATLAB: tempG1 = (1:length(G1))'.^(ones(size(G2)));
        # MATLAB: tempG2 = (1:length(G2)).^(ones(size(G1))');
        # This creates matrices where tempG1[i,j] = i+1 and tempG2[i,j] = j+1 (MATLAB 1-based)
        # For array indexing in Python, we need to convert to 0-based later
        tempG1 = np.tile(np.arange(1, len(G1) + 1).reshape(-1, 1), (1, len(G2)))
        tempG2 = np.tile(np.arange(1, len(G2) + 1).reshape(1, -1), (len(G1), 1))
        
        if isEqual:
            # we can ignore the left lower triangle in this case
            # as it's the same as the right upper triangle
            # -> double right upper triangle
            
                    # MATLAB: G1_ind = 1:length(G1);
            # MATLAB: G2_ind = 1:length(G2);
            # Convert to 0-based indexing for Python array access
            G1_ind = np.arange(0, len(G1))  # 0-based indexing for Python
            G2_ind = np.arange(0, len(G2))  # 0-based indexing for Python
            
            # MATLAB: G1_ind2 = reshape(triu(tempG1, 1)', 1, []);
            # MATLAB: G2_ind2 = reshape(triu(tempG2, 1)', 1, []);
            triu_G1 = np.triu(tempG1, k=1).T
            triu_G2 = np.triu(tempG2, k=1).T
            G1_ind2 = triu_G1.reshape(1, -1).flatten()
            G2_ind2 = triu_G2.reshape(1, -1).flatten()
            
            # MATLAB: G1_ind2 = G1_ind2(G1_ind2 > 0);
            # MATLAB: G2_ind2 = G2_ind2(G2_ind2 > 0);
            G1_ind2 = G1_ind2[G1_ind2 > 0]
            G2_ind2 = G2_ind2[G2_ind2 > 0]
            
            # Convert to 0-based indexing for Python array access
            G1_ind2 = G1_ind2 - 1
            G2_ind2 = G2_ind2 - 1
            
        else:
            # calculate all values
            # MATLAB: G1_ind = tempG1;
            # MATLAB: G2_ind = tempG2;
            G1_ind = tempG1
            G2_ind = tempG2
            
            # MATLAB: G1_ind = reshape(G1_ind, 1, []);
            # MATLAB: G2_ind = reshape(G2_ind, 1, []);
            G1_ind = G1_ind.reshape(1, -1)
            G2_ind = G2_ind.reshape(1, -1)
            
            # Convert to 0-based indexing for Python array access
            G1_ind = G1_ind - 1
            G2_ind = G2_ind - 1
    
    return G1_ind, G2_ind, G1_ind2, G2_ind2
