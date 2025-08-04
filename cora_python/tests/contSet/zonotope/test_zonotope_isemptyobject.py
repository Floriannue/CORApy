# test_zonotope_isemptyobject - unit test function of isemptyobject
#
# Syntax:
#    pytest cora_python/tests/contSet/zonotope/test_zonotope_isemptyobject.py
#
# Inputs:
#    -
#
# Outputs:
#    -
#
# Authors:       Mark Wetzlinger (MATLAB)
#                Python translation by AI Assistant
# Written:       03-June-2022 (MATLAB)
# Python translation: 2025

import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope

def test_zonotope_isemptyobject():
    Z1 = Zonotope.empty(2)
    Z2 = Zonotope(np.array([1, 1]))
    G3 = np.array([[1, 3, -2], [2, -4, 2]])
    Z3 = Zonotope(np.array([1, 1]), G3)
    assert Z1.is_empty()
    assert not Z2.is_empty()
    assert not Z3.is_empty() 