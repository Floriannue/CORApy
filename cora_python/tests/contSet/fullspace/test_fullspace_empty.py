"""
test_fullspace_empty - unit test function of empty

Syntax:
   res = test_fullspace_empty

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

def test_fullspace_empty():
    """Test the empty method of fullspace"""
    
    # create empty fullspace
    fs = Fullspace.empty()
    
    # check result
    assert fs.dimension == 0
    assert fs.isemptyobject()


# ------------------------------ END OF CODE ------------------------------ 