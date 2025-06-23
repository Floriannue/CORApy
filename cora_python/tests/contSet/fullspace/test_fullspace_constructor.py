"""
test_fullspace_constructor - unit test function of constructor

Syntax:
    res = test_fullspace_constructor()

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger
Written:       05-April-2023
Last update:   ---
Last revision: ---
"""

import pytest
from cora_python.contSet.fullspace import Fullspace
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_fullspace_constructor():
    """Test fullspace constructor"""
    
    # dimension greater or equal to one
    n = 2
    fs = Fullspace(n)
    assert fs.dimension == n
    
    # too many input arguments - expecting CORAerror
    with pytest.raises(CORAerror):
        Fullspace(n, n)


if __name__ == "__main__":
    test_fullspace_constructor() 