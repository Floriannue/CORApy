"""
generateRandom - generates a random fullspace

Syntax:
    fs = generateRandom()
    fs = generateRandom('Dimension', n)

Inputs:
    Name-Value pairs (all options, arbitrary order):
        <'Dimension', n> - dimension

Outputs:
    fs - random fullspace object

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
    from .fullspace import Fullspace


def generateRandom(**kwargs) -> 'Fullspace':
    """
    Generates a random fullspace
    
    Args:
        **kwargs: dimension (optional).
        
    Returns:
        fs: random fullspace object
    """
    
    from .fullspace import Fullspace
    
    # Default values (Python-style lowercase parameter names)
    listOfNameValuePairs = [
        'dimension', 2
    ]
    
    # Parse input arguments
    [n] = NVpairsPlainSetDefaultValues(listOfNameValuePairs, list(kwargs.keys()), list(kwargs.values()))
    
    # Check input arguments
    inputArgsCheck([
        [n, 'att', 'numeric', ['scalar', 'positive', 'integer']]
    ])
    
    return Fullspace(n) 