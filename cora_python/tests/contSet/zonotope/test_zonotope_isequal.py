# test_zonotope_isequal - unit test function of isequal
#
# Syntax:
#    pytest cora_python/tests/contSet/zonotope/test_zonotope_isequal.py
#
# Inputs:
#    -
#
# Outputs:
#    -
#
# Authors:       Mark Wetzlinger (MATLAB)
#                Python translation by AI Assistant
# Written:       17-September-2019 (MATLAB)
# Last update:   21-April-2020 (MATLAB), 09-August-2020 (MATLAB)
# Python translation: 2025

import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope

def test_zonotope_isequal():
    # 1D, minimal and non-minimal
    # Z1 = zonotope(4,[1 3 -2]);
    # Z2 = zonotope(4,6);
    Z1 = Zonotope(4, np.array([[1, 3, -2]]))
    Z2 = Zonotope(4, np.array([[6]]))
    assert Z1.isequal(Z2)
    assert Z2.isequal(Z1)

    # 2D, different order of generators
    # Z1 = zonotope(ones(2,1),[1 2 5 3 3; 2 3 0 4 1]);
    # Z2 = zonotope(ones(2,1),[2 1 3 5 3; 3 2 4 0 1]);
    Z1 = Zonotope(np.ones((2, 1)), np.array([[1, 2, 5, 3, 3], [2, 3, 0, 4, 1]]))
    Z2 = Zonotope(np.ones((2, 1)), np.array([[2, 1, 3, 5, 3], [3, 2, 4, 0, 1]]))
    assert Z1.isequal(Z2)
    assert Z2.isequal(Z1)

    # 2D different sign
    # Z1 = zonotope([0;1],[1 0 -1; 1 1 2]);
    # Z2 = zonotope([0;1],[-1 0 -1; -1 -1 2]);
    Z1 = Zonotope(np.array([[0], [1]]), np.array([[1, 0, -1], [1, 1, 2]]))
    Z2 = Zonotope(np.array([[0], [1]]), np.array([[-1, 0, -1], [-1, -1, 2]]))
    assert Z1.isequal(Z2)
    assert Z2.isequal(Z1)

    # 3D, with zero-generator
    # Z1 = zonotope([1;5;-1],[2 4; 6 0; 4 8]);
    # Z2 = zonotope([1;4;-1],[2 4; 6 0; 4 8]);
    # Z3 = zonotope([1;5;-1],[2 0 4; 6 0 0; 4 0 8]);
    Z1 = Zonotope(np.array([[1], [5], [-1]]), np.array([[2, 4], [6, 0], [4, 8]]))
    Z2 = Zonotope(np.array([[1], [4], [-1]]), np.array([[2, 4], [6, 0], [4, 8]]))
    Z3 = Zonotope(np.array([[1], [5], [-1]]), np.array([[2, 0, 4], [6, 0, 0], [4, 0, 8]]))
    assert Z1.isequal(Z3)
    assert Z3.isequal(Z1)
    assert not Z1.isequal(Z2)
    assert not Z2.isequal(Z1) 