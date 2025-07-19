"""
test_fullspace_eventFcn - unit test function of eventFcn

Syntax:
   res = test_fullspace_eventFcn

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

def test_fullspace_eventFcn():
    """Test the eventFcn method of fullspace"""

    # init fullspace
    n = 2
    fs = Fullspace(n)
    
    # test event function
    x = np.array([1, 2])
    direction = 1
    val, isterminal, direction_out = fs.eventFcn(x, direction)
    
    # check result
    assert len(val) == n
    assert len(isterminal) == n
    assert len(direction_out) == n


# ------------------------------ END OF CODE ------------------------------ 