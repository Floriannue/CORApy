"""
test_fullspace_Inf - unit test function of R^n instantiation

Syntax:
   res = test_fullspace_Inf

Inputs:
   -

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner
Written:       16-January-2024
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE -------------------------------
"""

import pytest
from cora_python.contSet.fullspace import Fullspace

def test_fullspace_Inf():
    """Test the Inf method of fullspace"""
    # 1D
    n = 1
    fs = Fullspace.Inf(n)
    assert fs.representsa('fullspace') and fs.dim() == 1
    
    # 5D
    n = 5
    fs = Fullspace.Inf(n)
    assert fs.representsa('fullspace') and fs.dim() == 5
    


# ------------------------------ END OF CODE ------------------------------ 