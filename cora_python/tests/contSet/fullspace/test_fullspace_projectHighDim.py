"""
test_fullspace_projectHighDim - unit test function of projectHighDim

Syntax:
   res = test_fullspace_projectHighDim

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

def test_fullspace_projectHighDim():
    """Test the projectHighDim method of fullspace"""
    
    # init fullspace
    n = 2
    fs = Fullspace(n)
    
    # project to higher dimension
    N = 4
    proj = [1, 2, 3, 4]
    fs_ = fs.projectHighDim_(N, proj)
    
    # check result
    assert fs_ == fs

# ------------------------------ END OF CODE ------------------------------ 