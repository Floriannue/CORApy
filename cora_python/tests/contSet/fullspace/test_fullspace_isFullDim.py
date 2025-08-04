"""
test_fullspace_isFullDim - unit test function of isFullDim

Syntax:
   res = test_fullspace_isFullDim

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

def test_fullspace_isFullDim():
    """Test the isFullDim method of fullspace"""
    # init fullspace
    n = 2
    fs = Fullspace(n)
    
    # check property
    assert fs.isFullDim()

# ------------------------------ END OF CODE ------------------------------ 