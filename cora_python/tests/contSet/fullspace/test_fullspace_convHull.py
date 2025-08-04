"""
test_fullspace_convHull - unit test function of convHull

Syntax:
   res = test_fullspace_convHull

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

def test_fullspace_convHull():
    """Test the convHull method of fullspace"""
    # init fullspace
    n = 2
    fs = Fullspace(n)
    
    # compute convex hull with itself (should return fullspace)
    fs_ = fs.convHull_(fs)
    
    # check result
    assert fs == fs_

# ------------------------------ END OF CODE ------------------------------ 