"""
combinator - Perform basic permutation and combination samplings

This function returns combinations without repetition for the set 1:N,
taken K at a time.

Syntax:
    A = combinator(N, K, 'c')

Inputs:
    N - positive integer (upper bound of set 1:N)
    K - non-negative integer (number of elements to choose)
    mode - sampling mode ('c' for combinations)

Outputs:
    A - matrix where each row contains one combination

Example:
    A = combinator(4, 2, 'c')
    # Returns [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

Authors: Matt Fig (MATLAB)
         Python translation by AI Assistant
Written: 5/30/2009 (MATLAB)
Python translation: 2025
"""

import numpy as np
from itertools import combinations


def combinator(N: int, K: int, mode: str = 'c') -> np.ndarray:
    """
    Compute combinations of the set 1:N taken K at a time
    
    Args:
        N: Upper bound of set (positive integer)
        K: Number of elements to choose (non-negative integer)  
        mode: Sampling mode ('c' for combinations)
        
    Returns:
        Array where each row contains one combination (1-indexed)
    """
    # Input validation
    if N <= 0 or not isinstance(N, int):
        raise ValueError('N should be one positive integer')
    if K < 0 or not isinstance(K, int):
        raise ValueError('K should be one non-negative integer')
    if K > N:
        raise ValueError('K must be less than or equal to N for combinations without repetition')
    
    # Handle edge cases
    if K == 0:
        return np.empty((1, 0), dtype=int)
    if N == 0:
        return np.empty((0, K), dtype=int)
    
    # Generate combinations using itertools (0-indexed)
    combos = list(combinations(range(1, N+1), K))
    
    # Convert to numpy array
    if len(combos) > 0:
        result = np.array(combos, dtype=int)
    else:
        result = np.empty((0, K), dtype=int)
    
    return result 