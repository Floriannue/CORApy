"""
test_zonotope_or_ - unit test function of or_

Syntax:
    res = test_zonotope_or_

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       26-October-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval


def test_zonotope_or():
    """Unit test function of or_ - mirrors MATLAB test_zonotope_or.m"""
    
    # check empty zonotope
    Z = Zonotope.empty(2)
    Zres = Z.or_(Z)
    
    # instantiate zonotope
    zono1 = Zonotope(np.array([[4, 2, 2], [1, 2, 0]]))
    zono2 = Zonotope(np.array([[3, 1, -1, 1], [3, 1, 2, 0]]))
    Zres = zono1.or_(zono2)
    
    # axis aligned zonotopes
    zono1 = Zonotope(Interval(np.array([2, 3]), np.array([3, 4])))
    zono2 = Zonotope(Interval(np.array([6, 3]), np.array([7, 4])))
    Zres = zono1.or_(zono2)
    
    # gather results
    res = True
    
    return res


if __name__ == "__main__":
    pytest.main([__file__]) 