"""
test_zonotope_zonotopeNorm - unit test function of zonotope norm

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-January-2024 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.zonotope.zonotopeNorm import zonotopeNorm


def test_zonotope_zonotopeNorm():
    """
    Unit test function for zonotopeNorm that matches MATLAB version exactly.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    
    # empty zonotopes
    Z = Zonotope.empty(2)
    assert zonotopeNorm(Z, np.array([1, -1]).reshape(-1, 1))[0] == np.inf
    assert zonotopeNorm(Z, np.empty((2, 0)))[0] == 0
    
    # 2D, only center
    Z = Zonotope(np.array([0, 0]))
    assert zonotopeNorm(Z, np.array([0, 0]).reshape(-1, 1))[0] == 0
    assert zonotopeNorm(Z, np.array([1, 0]).reshape(-1, 1))[0] == np.inf
    
    # 2D, center and generators
    c = np.array([0, 0])
    G = np.array([[1, -2, 2, 0], [-1, 1, 0, 1]])
    Z = Zonotope(c, G)
    
    # Test specific values from MATLAB
    assert abs(zonotopeNorm(Z, np.array([5, 3]).reshape(-1, 1))[0] - 2.2) < 1e-10
    assert abs(zonotopeNorm(Z, np.array([-5, 3]).reshape(-1, 1))[0] - 1.0) < 1e-10
    
    # shifted center does not influence the result
    Z = Z + np.array([1, -2])
    assert abs(zonotopeNorm(Z, np.array([5, 3]).reshape(-1, 1))[0] - 2.2) < 1e-10
    assert abs(zonotopeNorm(Z, np.array([-5, 3]).reshape(-1, 1))[0] - 1.0) < 1e-10
    
    # check boundary point - first case
    # In MATLAB: [~,p,beta_true] = supportFunc(Z,[1;1]);
    # We need to compute the support vector manually
    c = np.array([0, 0])
    G = np.array([[1, -2, 2, 0], [-1, 1, 0, 1]])
    Z = Zonotope(c, G)
    
    # Compute support vector in direction [1, 1]
    dir = np.array([1, 1])
    c_proj = dir @ c
    G_proj = dir @ G
    fac = np.sign(G_proj)  # factors for upper bound
    p_boundary = c.reshape(-1, 1) + G @ fac.reshape(-1, 1)
    
    res, minimizer = zonotopeNorm(Z, p_boundary)
    assert abs(res - 1.0) < 1e-10
    
    # Verify that minimizer satisfies the constraint: p = c + G * minimizer
    if minimizer.size > 0:
        p_reconstructed = c.reshape(-1, 1) + G @ minimizer.reshape(-1, 1)
        np.testing.assert_allclose(p_reconstructed.flatten(), p_boundary.flatten(), atol=1e-10)
    
    # Second case with different zonotope
    # In MATLAB: [~,p] = supportFunc(Z,[1;1]);
    c = np.array([0, 0])
    G = np.array([[1, 0, -1], [0, 1, -1]])
    Z = Zonotope(c, G)
    
    # Compute support vector in direction [1, 1]
    dir = np.array([1, 1])
    c_proj = dir @ c
    G_proj = dir @ G
    fac = np.sign(G_proj)  # factors for upper bound
    p_boundary2 = c.reshape(-1, 1) + G @ fac.reshape(-1, 1)
    
    res, minimizer = zonotopeNorm(Z, p_boundary2)
    assert abs(res - 1.0) < 1e-10
    
    # Verify constraint satisfaction
    if minimizer.size > 0:
        p_reconstructed = c.reshape(-1, 1) + G @ minimizer.reshape(-1, 1)
        np.testing.assert_allclose(p_reconstructed.flatten(), p_boundary2.flatten(), atol=1e-10)
    
    # combine results
    return True


if __name__ == '__main__':
    success = test_zonotope_zonotopeNorm()
    if success:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!") 