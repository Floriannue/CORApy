"""
test_fullspace_isBounded - unit test function of isBounded

Syntax:
   res = test_fullspace_isBounded

Inputs:
   -

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       16-October-2024
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE -------------------------------
"""

import pytest
from cora_python.contSet.fullspace import Fullspace

def test_fullspace_isBounded():
    """Test the isBounded method of fullspace"""
    # check boundedness
    fs = Fullspace(2)
    assert not fs.isBounded()
    
    # check boundedness of 0-dimensional set
    fs = Fullspace(0)
    assert fs.isBounded()  # as there is only one element in the set (zeros(0))

# ------------------------------ END OF CODE ------------------------------ 