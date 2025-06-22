import numpy as np # Keep numpy import if it's used elsewhere in the file
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def empty(n=0) -> 'EmptySet':
    """
    empty - instantiates an empty emptySet

    Syntax:
       O = emptySet.empty(n)

    Inputs:
       n - dimension

    Outputs:
       O - empty emptySet object

    Example:
       O = emptySet.empty(2);

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Mark Wetzlinger
    Written:       09-January-2024
    Last update:   15-January-2024 (TL, parse input)
    Last revision: ---
    """

    # Basic validation
    if not isinstance(n, (int, float)) or n < 0:
        raise CORAerror('CORA:wrongValue', 'first', 'Dimension must be a non-negative number')
    
    n = int(n)  # Convert to integer

    # call constructor (and check n there)
    from .emptySet import EmptySet
    O = EmptySet(n)
    return O 