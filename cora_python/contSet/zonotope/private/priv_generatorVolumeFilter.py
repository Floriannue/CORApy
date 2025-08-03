"""
priv_generatorVolumeFilter - filters out generators by finding the
    combinations returning the biggest volume

Syntax:
    Gred = priv_generatorVolumeFilter(G, rem)

Inputs:
    G - matrix of generators
    rem - number of remaining generators

Outputs:
    Gred - cell array of generators

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       12-September-2008 (MATLAB)
Last update:   19-July-2010 (MATLAB)
                14-March-2019 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import List


def priv_generatorVolumeFilter(G: np.ndarray, rem: int) -> List[np.ndarray]:
    """
    Private generator volume filter - filters out generators by finding the
    combinations returning the biggest volume
    
    Args:
        G: matrix of generators
        rem: number of remaining generators
        
    Returns:
        cell array of generators
    """
    # Determine generators by volume maximation:
    # possible combinations of n=dim generators from all generators
    rows, cols = G.shape
    
    # Create combinations
    from itertools import combinations
    comb = list(combinations(range(cols), rows))
    nrOfComb = len(comb)
    
    parallelogramVol = np.zeros(nrOfComb)
    for i in range(nrOfComb):
        try:
            # Obtain Gpicked
            Gpicked = G[:, list(comb[i])]
            parallelogramVol[i] = abs(np.linalg.det(Gpicked))
        except:
            parallelogramVol[i] = 0
            print('parallelogram volume could not be computed')
    
    # Obtain indices corresponding to the largest values (equivalent to MATLAB's maxk)
    index = np.argsort(parallelogramVol)[::-1][:rem]
    
    # Store the generator combinations in cells
    Gred = []
    for i in range(len(index)):
        generatorIndices = list(comb[index[i]])
        Gred.append(G[:, generatorIndices])
    
    return Gred 