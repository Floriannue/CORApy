# test_zonotope_isFullDim - unit test function of isFullDim
#
# Syntax:
#    pytest cora_python/tests/contSet/zonotope/test_zonotope_isFullDim.py
#
# Inputs:
#    -
#
# Outputs:
#    -
#
# Authors:       Mark Wetzlinger, Adrian Kulmburg (MATLAB)
#                Python translation by AI Assistant
# Written:       27-July-2021 (MATLAB)
# Last update:   04-February-2024 (MATLAB)
# Python translation: 2025

import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.zonotope.isFullDim import isFullDim

def test_zonotope_isFullDim():
    # empty case
    Z = Zonotope.empty(2)
    res_subspace, subspace = isFullDim(Z)
    assert not isFullDim(Z)[0]
    assert not res_subspace
    assert subspace is None or subspace.size == 0

    # full-rank matrix
    c = np.array([[0], [0]])
    G = np.array([[1, 2], [2, 1]])
    Z = Zonotope(c, G)
    res_subspace, subspace = isFullDim(Z)
    assert isFullDim(Z)[0]
    assert res_subspace
    assert subspace.shape[1] == 2

    # degenerate zonotope
    G = np.array([[1, 0], [0, 0]])
    Z = Zonotope(c, G)
    res_subspace, subspace = isFullDim(Z)
    true_subspace = np.array([[1], [0]])
    # Check if subspace contains true_subspace (up to numerical rank)
    same_subspace = np.linalg.matrix_rank(np.hstack([subspace, true_subspace]), 1e-6) == subspace.shape[1]
    assert not isFullDim(Z)[0]
    assert not res_subspace
    assert same_subspace

    # almost degenerate zonotope
    eps = 1e-8
    c = np.array([[0], [0]])
    G = np.array([[1, 1-eps], [1, 1]])
    Z = Zonotope(c, G)
    assert not isFullDim(Z)[0] 