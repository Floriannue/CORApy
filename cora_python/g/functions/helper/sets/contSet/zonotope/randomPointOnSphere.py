"""
randomPointOnSphere - generates a random vector

Syntax:
    x = randomPointOnSphere(n)

Inputs:
    n - dimension

Outputs:
    x - random vector

Example: 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2008 (MATLAB)
Last update:   ---
Python translation: 2025
"""

import numpy as np
from typing import Union


def randomPointOnSphere(n: Union[int, np.integer]) -> np.ndarray:
    """
    Generate a random point on the unit sphere in n dimensions.
    
    Args:
        n: Dimension of the space
        
    Returns:
        np.ndarray: Random point on the unit sphere (column vector)
    """
    # Generate random vector from standard normal distribution
    x = np.random.randn(n, 1)
    
    # Normalize to get a point on the unit sphere
    return x / np.linalg.norm(x) 