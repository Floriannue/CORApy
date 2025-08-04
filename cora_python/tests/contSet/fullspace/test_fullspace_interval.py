"""
test_fullspace_interval - unit test function of interval

Syntax:
   res = test_fullspace_interval

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
from cora_python.contSet.interval import Interval

def test_fullspace_interval():
    """Test the interval method of fullspace"""
    
    # init fullspace
    n = 3
    fs = Fullspace(n)
    
    # convert to interval
    I = fs.interval()
    
    # true result
    I_true = Interval(-np.inf * np.ones(n), np.inf * np.ones(n))
    
    # compare results
    assert I == I_true

# ------------------------------ END OF CODE ------------------------------      