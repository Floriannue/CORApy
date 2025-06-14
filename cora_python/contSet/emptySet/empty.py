import numpy as np # Keep numpy import if it's used elsewhere in the file
from cora_python.g.functions.matlab.validate.check import inputArgsCheck

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

    # parse input
    inputArgsCheck([[n, 'att', 'numeric', {'scalar', 'nonnegative'}]])

    # call constructor (and check n there)
    O = EmptySet(n)
    return O 