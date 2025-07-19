"""
test_fullspace_origin - unit test function of origin

Syntax:
   res = test_fullspace_origin

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
import numpy as np
from cora_python.contSet.fullspace import Fullspace

def test_fullspace_origin():
    """Test the origin method of fullspace"""
    
    # create origin fullspace
    n = 2
    fs = Fullspace.origin(n)
    
    # check result
    assert fs.dimension == n
    assert np.all(fs.center() == 0)
# ------------------------------ END OF CODE ------------------------------ 