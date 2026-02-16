"""
generateRandom - generates a random polynomial zonotope

Syntax:
    pZ = generateRandom()
    pZ = generateRandom('Dimension', n)
    pZ = generateRandom('Dimension', n, 'NrGenerators', nrGens)
    pZ = generateRandom('Dimension', n, 'NrGenerators', nrGens, 'NrIndepGenerators', nrIndepGens)
    pZ = generateRandom('Dimension', n, 'NrGenerators', nrGens, 'NrIndepGenerators', nrIndepGens, 'MaxDegree', maxDeg)

Inputs:
    Name-Value pairs (all options, arbitrary order):
        <'Dimension', n> - dimension
        <'NrGenerators', nrGens> - number of dependent generators
        <'NrIndepGenerators', nrIndepGens> - number of independent generators
        <'MaxDegree', maxDeg> - maximum degree of dependent generators

Outputs:
    pZ - random polyZonotope object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       26-March-2018 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.NVpairsPlainSetDefaultValues import NVpairsPlainSetDefaultValues

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope


def generateRandom(**kwargs) -> 'PolyZonotope':
    """
    Generates a random polynomial zonotope.

    Syntax (Python-style lowercase kwargs):
        pZ = PolyZonotope.generateRandom(dimension=2)
        pZ = PolyZonotope.generateRandom(dimension=2, nr_generators=5, nr_indep_generators=2)

    Args:
        **kwargs: dimension, nr_generators, nr_indep_generators, max_degree (all optional).

    Returns:
        PolyZonotope: random polynomial zonotope
    """
    
    from .polyZonotope import PolyZonotope
    
    # Default values (Python-style lowercase parameter names)
    listOfNameValuePairs = [
        'dimension', 2,
        'nr_generators', 5,
        'nr_indep_generators', 2,
        'max_degree', 3
    ]
    
    # Parse input arguments
    [n, nrGens, nrIndepGens, maxDeg] = NVpairsPlainSetDefaultValues(
        listOfNameValuePairs, list(kwargs.keys()), list(kwargs.values()))
    
    # Check input arguments
    inputArgsCheck([
        [n, 'att', 'numeric', ['scalar', 'positive', 'integer']],
        [nrGens, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']],
        [nrIndepGens, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']],
        [maxDeg, 'att', 'numeric', ['scalar', 'positive', 'integer']]
    ])
    
    # Generate random center
    c = -1 + 2 * np.random.rand(n, 1)
    
    # Generate random dependent generator matrix
    G = -1 + 2 * np.random.rand(n, nrGens)
    
    # Generate random independent generator matrix
    Grest = -1 + 2 * np.random.rand(n, nrIndepGens)
    
    # Generate random exponent matrix
    if nrGens > 0:
        # Create random exponent matrix with maximum degree constraint
        expMat = np.random.randint(0, maxDeg + 1, size=(nrGens, nrGens))
        # Ensure diagonal elements are non-zero to avoid trivial cases
        np.fill_diagonal(expMat, np.random.randint(1, maxDeg + 1, size=nrGens))
    else:
        expMat = np.zeros((0, 0), dtype=int)
    
    # Generate random identifier vector
    if nrGens > 0:
        id = np.arange(1, nrGens + 1)
    else:
        id = np.array([])
    
    return PolyZonotope(c, G, Grest, expMat, id) 