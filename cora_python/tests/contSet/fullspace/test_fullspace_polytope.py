"""
test_fullspace_polytope - unit test function of polytope conversion

Syntax:
   res = test_fullspace_polytope

Inputs:
   -

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       15-December-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE -------------------------------
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace import Fullspace

def test_fullspace_polytope():
    """Test the polytope method of fullspace"""
    # init fullspace
    fs = Fullspace(2)
    P = fs.polytope()
    assert not P.isBounded()
    assert P.supportFunc(np.array([1, 0]), 'upper') == np.inf
    assert P.supportFunc(np.array([-1, 0]), 'upper') == np.inf
    assert P.supportFunc(np.array([0, 1]), 'upper') == np.inf
    assert P.supportFunc(np.array([0, -1]), 'upper') == np.inf
    

# ------------------------------ END OF CODE ------------------------------ 