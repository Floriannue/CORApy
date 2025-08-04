"""
test_fullspace_generateRandom - unit test function of generateRandom

Syntax:
   res = test_fullspace_generateRandom

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

def test_fullspace_generateRandom():
    """Test the generateRandom method of fullspace"""
    
    # generate random fullspace
    fs = Fullspace.generateRandom()
    
    # check result
    assert isinstance(fs, Fullspace)
    assert fs.dimension > 0

# ------------------------------ END OF CODE ------------------------------ 