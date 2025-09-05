"""
dominantDirections - computes the directions that span a parallelotope
    which tightly encloses a zonotope Z

Syntax:
    S = dominantDirections(Z, varargin)

Inputs:
    Z - zonotope object
    filterLength1 - length of the length filter
    filterLength2 - length of the generator volume filter

Outputs:
    S - matrix containing the dominant directions as column vectors

Example:
    Z = Zonotope(np.array([[0], [0]]), -1 + 2 * np.random.rand(2, 20))
    S = dominantDirections(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-July-2010 (MATLAB)
Last update: --- (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from .dim import dim
from cora_python.g.functions.helper.sets.contSet.zonotope import nonzeroFilter
from .private import priv_lengthFilter, priv_generatorVolumeFilter, priv_volumeFilter


def dominantDirections(Z: Zonotope, filterLength1: Optional[int] = None, 
                      filterLength2: Optional[int] = None) -> np.ndarray:
    """
    Computes the directions that span a parallelotope which tightly encloses a zonotope Z
    
    Args:
        Z: zonotope object
        filterLength1: length of the length filter (default: n+5)
        filterLength2: length of the generator volume filter (default: n+3)
        
    Returns:
        Matrix containing the dominant directions as column vectors
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), -1 + 2*np.random.rand(2, 20))
        S = dominantDirections(Z)
    """
    # Dimension
    n = dim(Z)
    
    # Parse input arguments
    defaults = [n+5, n+3]
    args = [filterLength1, filterLength2]
    result = setDefaultValues(defaults, args)
    filterLength1, filterLength2 = result
    
    # Check input arguments
    inputArgsCheck([
        [Z, 'att', 'zonotope'],
        [filterLength1, 'att', 'numeric', ['nonnan', 'scalar', 'positive']],
        [filterLength2, 'att', 'numeric', ['nonnan', 'scalar', 'positive']]
    ])
    
    # Delete zero-generators
    if Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 'Generator matrix is None')
    G = nonzeroFilter(Z.G)
    
    # Number of generators
    nrOfGens = G.shape[1]
    
    # Correct filter length if necessary
    if filterLength1 > nrOfGens:
        filterLength1 = nrOfGens
    
    if filterLength2 > nrOfGens:
        filterLength2 = nrOfGens
    
    # Length filter
    G = priv_lengthFilter(G, filterLength1)
    
    # Apply generator volume filter
    Gcells = priv_generatorVolumeFilter(G, filterLength2)
    
    # Pick generator with the best volume
    G_picked = priv_volumeFilter(Gcells, Z)
    
    # Select dominant directions S
    S = G_picked[0][:, :n]
    
    return S 