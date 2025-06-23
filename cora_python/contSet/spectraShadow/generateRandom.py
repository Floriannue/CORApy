"""
generateRandom - generates a random spectrahedral shadow

Syntax:
    sS = generateRandom()
    sS = generateRandom('Dimension', n)
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.NVpairsPlainSetDefaultValues import NVpairsPlainSetDefaultValues
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow


def generateRandom(**kwargs) -> SpectraShadow:
    """Generates a random spectrahedral shadow"""
    
    # Default values
    listOfNameValuePairs = ['Dimension', 2]
    
    # Parse input arguments
    [n] = NVpairsPlainSetDefaultValues(listOfNameValuePairs, list(kwargs.keys()), list(kwargs.values()))
    
    # Check input arguments
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'positive', 'integer']]])
    
    # Generate random spectrahedral shadow
    A = np.random.rand(3, 3)  # Random 3x3 matrix
    c = -1 + 2 * np.random.rand(n, 1)
    G = -1 + 2 * np.random.rand(n, 2)
    
    return SpectraShadow(A, c, G) 