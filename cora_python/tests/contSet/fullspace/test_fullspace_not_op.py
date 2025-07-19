"""
test_fullspace_not_op - unit test function of not_op

Syntax:
   res = test_fullspace_not_op

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
from cora_python.contSet.emptySet import EmptySet

def test_fullspace_not_op():
    """Test the not_op method of fullspace"""
    
    # init fullspace
    n = 2
    fs = Fullspace(n)
    
    # compute complement
    fs_ = ~fs
    
    # check result
    assert isinstance(fs_, EmptySet)

# ------------------------------ END OF CODE ------------------------------ 