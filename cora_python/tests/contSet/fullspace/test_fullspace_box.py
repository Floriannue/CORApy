"""
test_fullspace_box - unit test function of box

Syntax:
   res = test_fullspace_box

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

def test_fullspace_box():
    """Test the box method of fullspace"""
    # init fullspace
    n = 2
    fs = Fullspace(n)
    
    # compute box
    fs_ = fs.box()
    
    # compare results
    assert fs == fs_

# ------------------------------ END OF CODE ------------------------------ 