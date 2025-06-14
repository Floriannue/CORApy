import numpy as np
from typing import List, Any

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def checkNameValuePairs(NVpairs: List[Any], fullList: List[str]) -> bool:
    """
    checkNameValuePairs - checks if every name from a given list of
    name-value pairs is part of the full list of admissible name-value
    pairs (check is case-insensitive and works for chars and strings)

    Syntax:
       checkNameValuePairs(NVpairs,fullList)

    Inputs:
       NVpairs - list: list of given name-value pairs
       fullList - list: list of all potential name-value pairs

    Outputs:
       res = true/false

    Example:
       # NVpairs = ['Splits',8,'Order',5];
       # checkNameValuePairs(NVpairs,['Splits','Order']);

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: readNameValuePairs

    Authors:       Mark Wetzlinger
    Written:       08-June-2022
    Last update:   ---
    Last revision: ---
    """

    # default result
    res = True

    # number of given name-value pairs
    nrNVpairs = len(NVpairs) // 2

    # read list of name of given name-value pairs
    names = [NVpairs[i] for i in range(0, len(NVpairs), 2)]

    # go through all given names and check whether for containment in full list
    for i in range(nrNVpairs):
        if names[i].lower() not in [x.lower() for x in fullList]:
            raise CORAerror('CORA:redundantNameValuePair', names[i], fullList)
    return res 