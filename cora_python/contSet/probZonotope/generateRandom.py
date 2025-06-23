"""
generateRandom - generates a random probabilistic zonotope

Syntax:
    pZ = generateRandom()
    pZ = generateRandom('Dimension', n)
    pZ = generateRandom('Dimension', n, 'NrGenerators', nrGens)
    pZ = generateRandom('Dimension', n, 'NrGenerators', nrGens, 'NrProbGenerators', nrProbGens)

Inputs:
    Name-Value pairs (all options, arbitrary order):
        <'Dimension', n> - dimension
        <'NrGenerators', nrGens> - number of deterministic generators
        <'NrProbGenerators', nrProbGens> - number of probabilistic generators

Outputs:
    pZ - random probZonotope object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       27-September-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.NVpairsPlainSetDefaultValues import NVpairsPlainSetDefaultValues
from cora_python.contSet.probZonotope.probZonotope import ProbZonotope
from cora_python.contSet.zonotope.zonotope import Zonotope

if TYPE_CHECKING:
    pass


def generateRandom(**kwargs) -> ProbZonotope:
    """
    Generates a random probabilistic zonotope
    
    Args:
        **kwargs: Name-value pair arguments:
            Dimension: dimension (default: 2)
            NrGenerators: number of deterministic generators (default: 5)
            NrProbGenerators: number of probabilistic generators (default: 2)
        
    Returns:
        pZ: random probZonotope object
    """
    
    # Default values
    listOfNameValuePairs = [
        'Dimension', 2,
        'NrGenerators', 5,
        'NrProbGenerators', 2
    ]
    
    # Parse input arguments
    [n, nrGens, nrProbGens] = NVpairsPlainSetDefaultValues(
        listOfNameValuePairs, list(kwargs.keys()), list(kwargs.values()))
    
    # Check input arguments
    inputArgsCheck([
        [n, 'att', 'numeric', ['scalar', 'positive', 'integer']],
        [nrGens, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']],
        [nrProbGens, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']]
    ])
    
    # Generate random zonotope
    Z = Zonotope.generateRandom(Dimension=n, NrGenerators=nrGens)
    
    # Generate random probabilistic generators
    g = -1 + 2 * np.random.rand(n, nrProbGens)
    
    # Default gamma value
    gamma = 2
    
    return ProbZonotope(Z, g, gamma) 