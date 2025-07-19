"""
test_fullspace_lift - unit test function of lift

Syntax:
   res = test_fullspace_lift

Inputs:
   -

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger
Written:       06-April-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE -------------------------------
"""

import pytest
from cora_python.contSet.fullspace import Fullspace
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_fullspace_lift():
    """Test the lift method of fullspace"""
    # init fullspace
    n = 4
    fs = Fullspace(n)
    
    # higher-dimensional space
    N = 6
    dims = [1, 2, 5, 6]
    fs_ = fs.lift(N, dims)
    fs_true = Fullspace(N)
    assert fs_true == fs_
    
    # dimensions out of range
    projDims = [-1, 2, 3, 5]
    with pytest.raises(CORAerror, match='Wrong value'):
        fs.lift(N, projDims)
    
    # higher-dimensional space smaller than original space
    N = 3
    projDims = [1, 2, 5, 6]
    with pytest.raises(CORAerror, match='Wrong value'):
        fs.lift(N, projDims)

# ------------------------------ END OF CODE ------------------------------ 