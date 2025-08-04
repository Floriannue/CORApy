"""
test_fullspace_representsa - unit test function of representsa

Syntax:
   res = test_fullspace_representsa

Inputs:
   -

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       25-July-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE -------------------------------
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace import Fullspace
from cora_python.contSet.interval import Interval

def test_fullspace_representsa():
    """Test the representsa method of fullspace"""
    # init empty set
    n = 2
    O = Fullspace(n)
    
    # compare to other representations
    assert not O.representsa('origin')
    assert not O.representsa('point')
    assert not O.representsa('emptySet')
    assert not O.representsa('zonotope')
    
    isInterval, I = O.representsa('interval')
    assert isInterval
    assert I == Interval(-np.inf * np.ones(n), np.inf * np.ones(n))
    

# ------------------------------ END OF CODE ------------------------------ 