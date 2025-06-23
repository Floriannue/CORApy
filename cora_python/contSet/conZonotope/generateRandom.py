"""
generateRandom - generates a random constrained zonotope

Syntax:
    cZ = generateRandom()
    cZ = generateRandom('Dimension', n)
    cZ = generateRandom('Dimension', n, 'NrGenerators', nrGens)
    cZ = generateRandom('Dimension', n, 'NrGenerators', nrGens, 'NrConstraints', nrCon)

Inputs:
    Name-Value pairs (all options, arbitrary order):
        <'Dimension', n> - dimension
        <'NrGenerators', nrGens> - number of generators
        <'NrConstraints', nrCon> - number of constraints

Outputs:
    cZ - random conZonotope object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       27-September-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Optional

from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.preprocessing.NVpairsPlainSetDefaultValues import NVpairsPlainSetDefaultValues

if TYPE_CHECKING:
    from .conZonotope import ConZonotope


def generateRandom(**kwargs) -> 'ConZonotope':
    """
    Generates a random constrained zonotope
    
    Args:
        **kwargs: Name-value pair arguments:
            Dimension: dimension (default: 2)
            NrGenerators: number of generators (default: 5)
            NrConstraints: number of constraints (default: 2)
        
    Returns:
        cZ: random conZonotope object
    """
    
    from .conZonotope import ConZonotope
    
    # Default values
    listOfNameValuePairs = [
        'Dimension', 2,
        'NrGenerators', 5, 
        'NrConstraints', 2
    ]
    
    # Parse input arguments
    [n, nrGens, nrCon] = NVpairsPlainSetDefaultValues(listOfNameValuePairs, list(kwargs.keys()), list(kwargs.values()))
    
    # Check input arguments
    inputArgsCheck([
        [n, 'att', 'numeric', ['scalar', 'positive', 'integer']],
        [nrGens, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']],
        [nrCon, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']]
    ])
    
    # Generate random center
    c = -1 + 2 * np.random.rand(n, 1)
    
    # Generate random generator matrix
    G = -1 + 2 * np.random.rand(n, nrGens)
    
    # Generate random constraint matrix and vector
    if nrCon > 0 and nrGens > 0:
        A = -1 + 2 * np.random.rand(nrCon, nrGens)
        b = -1 + 2 * np.random.rand(nrCon, 1)
    else:
        A = np.zeros((0, nrGens))
        b = np.zeros((0, 1))
    
    return ConZonotope(c, G, A, b) 