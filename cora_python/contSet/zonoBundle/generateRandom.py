"""
generateRandom - generates a random zonotope bundle

Syntax:
    zB = generateRandom()
    zB = generateRandom('Dimension', n)
    zB = generateRandom('Dimension', n, 'NrZonotopes', nrZonos)
    zB = generateRandom('Dimension', n, 'NrZonotopes', nrZonos, 'NrGenerators', nrGens)

Inputs:
    Name-Value pairs (all options, arbitrary order):
        <'Dimension', n> - dimension
        <'NrZonotopes', nrZonos> - number of zonotopes in bundle
        <'NrGenerators', nrGens> - number of generators per zonotope

Outputs:
    zB - random zonoBundle object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       27-September-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.NVpairsPlainSetDefaultValues import NVpairsPlainSetDefaultValues

if TYPE_CHECKING:
    from .zonoBundle import ZonoBundle


def generateRandom(**kwargs) -> 'ZonoBundle':
    """
    Generates a random zonotope bundle
    
    Args:
        **kwargs: dimension, nr_zonotopes, nr_generators (all optional).
        
    Returns:
        zB: random zonoBundle object
    """
    
    from .zonoBundle import ZonoBundle
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # Default values (Python-style lowercase parameter names)
    listOfNameValuePairs = [
        'dimension', 2,
        'nr_zonotopes', 3,
        'nr_generators', 5
    ]
    
    # Parse input arguments
    [n, nrZonos, nrGens] = NVpairsPlainSetDefaultValues(
        listOfNameValuePairs, list(kwargs.keys()), list(kwargs.values()))
    
    # Check input arguments
    inputArgsCheck([
        [n, 'att', 'numeric', ['scalar', 'positive', 'integer']],
        [nrZonos, 'att', 'numeric', ['scalar', 'positive', 'integer']],
        [nrGens, 'att', 'numeric', ['scalar', 'nonnegative', 'integer']]
    ])
    
    # Generate random zonotopes
    zonotopes = []
    for i in range(nrZonos):
        zono = Zonotope.generateRandom(dimension=n, nr_generators=nrGens)
        zonotopes.append(zono)
    
    return ZonoBundle(zonotopes) 