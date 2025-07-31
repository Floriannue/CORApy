"""
test_zonotope_rank - unit test function of rank

Syntax:
    res = test_zonotope_rank

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Matthias Althoff
Written:       26-July-2016
Last update:   15-September-2019
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope


def test_zonotope_rank():
    """
    Unit test function of rank
    """
    # 2D zonotope
    Z = Zonotope(np.array([[1, 2, 0, 4], [5, 6, 0, 0], [-1, 4, 0, 8]]))
    assert Z.rank() == 2
    
    # empty zonotope
    Z_empty = Zonotope.empty(2)
    assert Z_empty.rank() == 0
    
    return True


if __name__ == "__main__":
    test_zonotope_rank()
    print("MATLAB test case passed!") 