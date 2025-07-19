"""
test_fullspace_project - unit test function of project

Syntax:
   res = test_fullspace_project

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
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE -------------------------------
"""

import pytest
from cora_python.contSet.fullspace import Fullspace
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_fullspace_project():
    """Test the project method of fullspace"""
    # init fullspace
    n = 4
    fs = Fullspace(n)
    
    # project to subspace
    projDims = [1, 4]
    fs_ = fs.project(projDims)
    fs_true = Fullspace(len(projDims))
    assert fs_true == fs_
    
    # subspace out of range
    projDims = [-1, 2]
    with pytest.raises(CORAerror):
        fs.project(projDims)
    
    # subspace out of range
    projDims = [3, 5]
    with pytest.raises(CORAerror):
        fs.project(projDims)
    

# ------------------------------ END OF CODE ------------------------------ 