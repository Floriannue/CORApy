"""
test_zonotope_mtimes - unit test function of mtimes

Syntax:
    res = test_zonotope_mtimes

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Mark Wetzlinger
Written:       26-July-2016
Last update:   05-October-2024 (MW, test projections)
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.contSet.representsa import representsa


def test_zonotope_mtimes():
    """Unit test function of mtimes - mirrors MATLAB test_zonotope_mtimes.m"""
    
    # 2D, matrix * empty zonotope
    Z = Zonotope.empty(2)
    
    # ...square matrix
    M = np.array([[-1, 2], [3, -4]])
    Z_mtimes = M * Z
    assert representsa(Z_mtimes, 'emptySet') and Z_mtimes.dim() == 2
    
    # ...projection onto subspace
    M = np.array([[-1, 2]])
    Z_mtimes = M * Z
    assert representsa(Z_mtimes, 'emptySet') and Z_mtimes.dim() == 1
    
    # ...projection to higher-dimensional space
    M = np.array([[-1, 2], [3, -4], [5, 6]])
    Z_mtimes = M * Z
    assert representsa(Z_mtimes, 'emptySet') and Z_mtimes.dim() == 3
    
    # 2D, matrix * non-empty zonotope
    Z = Zonotope(np.array([[-4], [1]]), np.array([[-3, -2, -1], [2, 3, 4]]))
    
    # ...square matrix
    M = np.array([[-1, 2], [3, -4]])
    Z_mtimes = M * Z
    Z_true = Zonotope(np.array([[6], [-16]]), np.array([[7, 8, 9], [-17, -18, -19]]))
    assert np.allclose(Z_mtimes.c, Z_true.c) and np.allclose(Z_mtimes.G, Z_true.G)
    
    # ...projection onto subspace
    M = np.array([[-1, 2]])
    Z_mtimes = M * Z
    Z_true = Zonotope(np.array([[6]]), np.array([[7, 8, 9]]))
    assert np.allclose(Z_mtimes.c, Z_true.c) and np.allclose(Z_mtimes.G, Z_true.G)
    
    # ...projection to higher-dimensional space
    M = np.array([[-1, 2], [1, 0], [0, 1]])
    Z_mtimes = M * Z
    Z_true = Zonotope(np.array([[6], [-4], [1]]), np.array([[7, 8, 9], [-3, -2, -1], [2, 3, 4]]))
    assert np.allclose(Z_mtimes.c, Z_true.c) and np.allclose(Z_mtimes.G, Z_true.G)
    
    # 2D, non-empty zonotope * matrix/scalar
    Z = Zonotope(np.array([[-4], [1]]), np.array([[-3, -2, -1], [2, 3, 4]]))
    M = 2
    Z_mtimes = Z * M
    Z_true = Zonotope(np.array([[-8], [2]]), np.array([[-6, -4, -2], [4, 6, 8]]))
    assert np.allclose(Z_mtimes.c, Z_true.c) and np.allclose(Z_mtimes.G, Z_true.G)
    
    # test completed
    return True


if __name__ == "__main__":
    pytest.main([__file__]) 