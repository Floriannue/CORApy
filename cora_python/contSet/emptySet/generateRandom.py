import numpy as np

from cora_python.g.functions.matlab.validate.check import checkNameValuePairs
from cora_python.g.functions.matlab.validate.preprocessing import readNameValuePair
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

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

    # name-value pair input
    # Use checkNameValuePairs (translated) and readNameValuePair (translated)
    opt = {}
    opt['Dimension'] = 0

    # parse input
    # MATLAB's checkNameValuePairs and readNameValuePair need to be properly used here
    # For now, a simplified direct assignment for 'Dimension'
    dimension = None
    NVpairs = list(varargin) # Convert tuple to list for modification

    # Check name-value pairs using the translated function
    # Assuming fullList is dynamically generated or defined elsewhere
    # For now, hardcode relevant names for generateRandom
    checkNameValuePairs(NVpairs, ['Dimension'])

    NVpairs, dimension = readNameValuePair(NVpairs, 'Dimension', 'numeric', dimension)

    if dimension is None:
        dimension = np.random.randint(1, 10)  # Random dimension between 1 and 9

    return EmptySet(dimension) 