"""
test_zonoBundle_representsa - unit test function of representsa

TRANSLATED FROM: cora_matlab/unitTests/contSet/zonoBundle/test_zonoBundle_representsa.m

Syntax:
    pytest cora_python/tests/contSet/zonoBundle/test_zonoBundle_representsa.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       23-April-2023 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: 20-July-2023 (MW, rename '...representsa', MATLAB)
"""

import pytest
import numpy as np
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.contSet.representsa import representsa


def test_zonoBundle_representsa_empty():
    """
    TRANSLATED TEST - Fully-empty zonoBundle represents emptySet
    """
    # fully-empty zonoBundle
    zB = ZonoBundle.empty(2)
    assert representsa(zB, 'emptySet')


def test_zonoBundle_representsa_non_empty():
    """
    TRANSLATED TEST - Non-empty intersection does not represent emptySet
    """
    # non-empty intersection
    Z1 = Zonotope(np.array([[1], [1]]), np.array([[3, 0], [0, 2]]))
    Z2 = Zonotope(np.array([[0], [0]]), np.array([[2, 2], [2, -2]]))
    zB = ZonoBundle([Z1, Z2])
    assert not representsa(zB, 'emptySet')


def test_zonoBundle_representsa_empty_intersection():
    """
    TRANSLATED TEST - Empty intersection represents emptySet
    """
    # empty intersection
    Z1 = Zonotope(np.array([[1], [1]]), np.array([[3, 0], [0, 2]]))
    Z2 = Zonotope(np.array([[-4], [1]]), np.array([[0.5, 1], [1, -1]]))
    zB = ZonoBundle([Z1, Z2])
    assert representsa(zB, 'emptySet')


if __name__ == "__main__":
    pytest.main([__file__])

