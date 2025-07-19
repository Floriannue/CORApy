"""
test_fullspace_volume - unit test function of volume

Syntax:
   res = test_fullspace_volume

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

def test_fullspace_volume():
    """Test the volume method of fullspace"""

    # init fullspace
    n = 2
    fs = Fullspace(n)
    
    # compute volume
    val = fs.volume()
    
    # check result
    assert val == np.inf


# ------------------------------ END OF CODE ------------------------------ 