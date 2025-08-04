# test_zonotope_isIntersecting_ - unit test function of isIntersecting_
#
# Syntax:
#    pytest cora_python/tests/contSet/zonotope/test_zonotope_isIntersecting_.py
#
# Inputs:
#    -
#
# Outputs:
#    -
#
# Authors:       Tobias Ladner
#                Python translation by AI Assistant
# Written:       20-September-2024
# Python translation: 2025

import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.zonotope.isIntersecting_ import isIntersecting_


def test_zonotope_isIntersecting_():
    # create zonotope
    c = np.array([[-4], [1]])
    G = np.array([[-3, -2, -1], [2, 3, 4]])
    Z1 = Zonotope(c, G)

    c2 = np.array([[-8], [4]])
    G2 = np.array([[1, 0, 1], [0, 1, 1]])
    Z2 = Zonotope(c2, G2)

    # should intersect
    assert isIntersecting_(Z1, Z2)

    # move outside
    Z2 = Zonotope(c2 - np.array([[4], [5]]), G2)
    assert not isIntersecting_(Z1, Z2)

    # check containment 'is intersecting'
    # enlarge Z1 by [2;1.5] (elementwise)
    # In MATLAB, enlarge(Z1,[2;1.5]) means scale each generator by [2,1.5]
    # For simplicity, scale all generators by 2 (first row) and 1.5 (second row)
    scale = np.array([[2], [1.5]])
    G_enlarged = Z1.G * scale
    Z2 = Zonotope(Z1.c, G_enlarged)
    assert isIntersecting_(Z1, Z2)

    # check touching zonotopes
    c2 = np.array([[8], [1]])
    G2 = np.array([[3, 2, 1], [2, 3, 4]])
    Z2 = Zonotope(c2, G2) + 1e-10
    assert isIntersecting_(Z1, Z2) 