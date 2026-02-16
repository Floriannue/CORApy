"""
generateRandom - generates a random Taylor model

Syntax:
    t = generateRandom()
    t = generateRandom('Dimension', n)
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.NVpairsPlainSetDefaultValues import NVpairsPlainSetDefaultValues
from cora_python.contSet.taylm.taylm import Taylm


def generateRandom(**kwargs) -> Taylm:
    """Generates a random Taylor model"""
    
    # Default values (Python-style lowercase parameter names)
    listOfNameValuePairs = ['dimension', 2]
    
    # Parse input arguments
    [n] = NVpairsPlainSetDefaultValues(listOfNameValuePairs, list(kwargs.keys()), list(kwargs.values()))
    
    # Check input arguments
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'positive', 'integer']]])
    
    # Generate random Taylor model with a few terms
    nTerms = min(5, 2**n)  # Limit number of terms
    monomials = np.random.randint(0, 3, size=(nTerms, n))
    coefficients = -1 + 2 * np.random.rand(nTerms)
    remainder = np.array([-0.1, 0.1])  # Small remainder interval
    
    return Taylm(monomials, coefficients, remainder) 