import numpy as np
from typing import TYPE_CHECKING

from cora_python.g.functions.matlab.validate.check import checkNameValuePairs
from cora_python.g.functions.matlab.validate.preprocessing import readNameValuePair
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .emptySet import EmptySet

def generateRandom(*varargin) -> 'EmptySet':
    """
    generateRandom - generates a random empty set

    Syntax:
       O = generateRandom()
       O = generateRandom('Dimension',n)

    Inputs:
       Name-Value pairs (all options, arbitrary order):
          <'Dimension',n> - dimension

    Outputs:
       O - random emptySet

    Example:
       O = emptySet.generateRandom();

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Mark Wetzlinger
    Written:       29-March-2019
    Last update:   09-January-2024 (TL, parse input)
                   03-March-2025 (TL, rework using readNameValuePair)
    Last revision: ---
    """

    # parse input arguments
    dimension = None
    
    # Process name-value pairs
    if len(varargin) > 0:
        # Check if we have name-value pairs
        if len(varargin) % 2 != 0:
            raise CORAerror('CORA:evenNumberInputArgs')
        
        # Process pairs
        for i in range(0, len(varargin), 2):
            name = varargin[i]
            value = varargin[i + 1]
            
            if name == 'Dimension':
                # Basic validation - check if it's a number
                if not isinstance(value, (int, float)) or value < 0:
                    raise CORAerror('CORA:wrongValue', 'third', 'Dimension must be a non-negative number')
                dimension = int(value)
            else:
                raise CORAerror('CORA:wrongNameValuePair', f'Unknown name-value pair: {name}')

    # Default dimension if not specified
    if dimension is None:
        dimension = np.random.randint(1, 10)  # Random dimension between 1 and 9

    # Import EmptySet here to avoid circular imports
    from .emptySet import EmptySet
    return EmptySet(dimension) 