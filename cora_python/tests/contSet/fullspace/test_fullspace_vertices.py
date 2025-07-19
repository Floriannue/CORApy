"""
test_fullspace_vertices - unit test function of vertices

Syntax:
   res = test_fullspace_vertices

Inputs:
   -

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger
Written:       25-April-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE -------------------------------
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace import Fullspace

def test_fullspace_vertices():
    """Test the vertices method of fullspace"""
    
    # init fullspace
    n = 2
    fs = Fullspace(n)
    
    # compute vertices
    V = fs.vertices()
    
    # check result
    assert isinstance(V, np.ndarray)
    # Check that all vertices are contained in the fullspace
    for i in range(V.shape[1]):
        res, _, _ = fs.contains_(V[:, i])
        assert res

# ------------------------------ END OF CODE ------------------------------ 