"""
test_zonotope_quadMap - unit test function of quadMap

Syntax:
    python -m pytest test_zonotope_quadMap.py

Inputs:
    -

Outputs:
    test results

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 26-July-2016 (MATLAB)
Last update: ---
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


def test_zonotope_quadMap():
    """Unit test function of quadMap - mirrors MATLAB test_zonotope_quadMap.m"""
    
    # create zonotopes
    Z1 = Zonotope(np.array([-4, 1]), np.array([[-3, -2], [2, 3]]))
    Z2 = Zonotope(np.array([1, -3]), np.array([[4, 2], [2, -1]]))
    
    # create matrices
    Q = [np.array([[1, -2], [-3, 4]]), np.array([[0.5, 0], [2, -1]])]
    
    # 1. quadMapSingle: Z1*Q*Z1 -----------------------------------------------
    
    # obtain result
    Zres = Z1.quadMap(Q)
    
    # obtain center and generator matrix
    c = Zres.c
    G = Zres.G
    
    # true result
    true_c = np.array([[102.5], [-16.25]])
    true_G = np.array([[27.5, 35, 95, 110, 125], [-5.75, -9.5, -14, -26, -32]])
    
    # compare solutions
    assert np.allclose(c, true_c)
    assert np.allclose(G, true_G)
    
    # 2. quadMapMixed: Z1*Q*Z2 ------------------------------------------------
    
    # obtain result
    Zres = Z1.quadMap(Z2, Q)
    
    # obtain center and generator matrix
    c = Zres.c
    G = Zres.G
    
    # true result
    true_c = np.array([[-43], [3]])
    true_G = np.array([[-51, -59, -4, -8, -12, -26, -32, -38], [8.5, 14, -2, 6, 14, 1, 7, 13]])
    
    # compare solutions
    assert np.allclose(c, true_c)
    assert np.allclose(G, true_G)


if __name__ == "__main__":
    test_zonotope_quadMap()
    print("All zonotope quadMap tests passed!") 