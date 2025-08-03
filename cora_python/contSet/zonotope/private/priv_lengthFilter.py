"""
priv_lengthFilter - filters out short generators

Syntax:
    Gred = priv_lengthFilter(G, rem)

Inputs:
    G - matrix of generators
    rem - number of remaining generators

Outputs:
    Gred - reduced matrix of generators

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       12-September-2008 (MATLAB)
Last update:   14-March-2019 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np


def priv_lengthFilter(G: np.ndarray, rem: int) -> np.ndarray:
    """
    Private length filter - filters out short generators
    
    Args:
        G: matrix of generators
        rem: number of remaining generators
        
    Returns:
        reduced matrix of generators
    """
    # Pre-filter generators
    h = np.linalg.norm(G, axis=0)
    
    # Choose largest values (equivalent to MATLAB's maxk)
    index = np.argsort(h)[::-1][:rem]
    Gred = G[:, index]
    
    return Gred 