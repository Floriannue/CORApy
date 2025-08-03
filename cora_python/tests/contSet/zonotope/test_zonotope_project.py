"""
test_zonotope_project - unit test function of project

Syntax:
    python -m pytest test_zonotope_project.py

Inputs:
    -

Outputs:
    test results

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 26-July-2016 (MATLAB)
Last update: 09-August-2020 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


def test_zonotope_project():
    """Unit test function of project - mirrors MATLAB test_zonotope_project.m"""
    
    # create zonotope
    Z = Zonotope(np.array([-4, 1, 5]), np.array([[-3, -2, -1], [2, 3, 4], [5, 5, 5]]))
    
    # project zonotope
    Z1 = Z.project([0, 2])  # MATLAB: [1 3] -> Python: [0, 2]
    c1 = Z1.c
    G1 = Z1.G
    
    # logical indexing
    Z2 = Z.project([True, False, True])
    c2 = Z2.c
    G2 = Z2.G
    
    # true result
    true_c = np.array([-4, 5])
    true_G = np.array([[-3, -2, -1], [5, 5, 5]])
    
    # check result
    assert np.allclose(c1.flatten(), true_c)
    assert np.allclose(G1, true_G)
    assert np.allclose(c2.flatten(), true_c)
    assert np.allclose(G2, true_G)


if __name__ == "__main__":
    test_zonotope_project()
    print("All zonotope project tests passed!") 