"""
test_fullspace_enclosePoints - unit test function of enclosePoints

Syntax:
   res = test_fullspace_enclosePoints

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

def test_fullspace_enclosePoints():
    """Test the enclosePoints method of fullspace"""
    
    # points
    p = np.array([[2, -4, 1], [-4, 3, 2], [0, 1, 9]])
    
    # enclose by R^n
    fs = Fullspace.enclosePoints(p)
    
    # check result
    assert fs.dim() == 3

# ------------------------------ END OF CODE ------------------------------ 