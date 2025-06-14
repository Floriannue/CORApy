import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope


def test_zonotope_abs():
    """
    Test abs method for zonotope - returns a zonotope with absolute values 
    of the center and the generators according to manual Appendix A.1.
    """
    
    # Test empty zonotope
    Z_empty = Zonotope.empty(2)
    Z_abs = abs(Z_empty)
    assert Z_abs.representsa_('emptySet')
    assert Z_abs.dim() == 2
    
    # Test 1D zonotope with positive center and generators
    Z = Zonotope(np.array([[2]]), np.array([[1, 0.5]]))
    Z_abs = abs(Z)
    # Should remain unchanged for positive values
    assert np.allclose(Z_abs.c, Z.c)
    assert np.allclose(Z_abs.G, Z.G)
    
    # Test 1D zonotope with negative center
    Z = Zonotope(np.array([[-2]]), np.array([[1, -0.5]]))
    Z_abs = abs(Z)
    expected_c = np.array([[2]])
    expected_G = np.array([[1, 0.5]])
    assert np.allclose(Z_abs.c, expected_c)
    assert np.allclose(Z_abs.G, expected_G)
    
    # Test 2D zonotope with mixed positive/negative values
    c = np.array([[1], [-2]])
    G = np.array([[2, -1, 0.5], [-3, 1, -0.8]])
    Z = Zonotope(c, G)
    Z_abs = abs(Z)
    
    expected_c = np.array([[1], [2]])
    expected_G = np.array([[2, 1, 0.5], [3, 1, 0.8]])
    assert np.allclose(Z_abs.c, expected_c)
    assert np.allclose(Z_abs.G, expected_G)
    
    # Test 3D zonotope
    c = np.array([[-1], [2], [-3]])
    G = np.array([[1, -2, 3], [-1, 0, 2], [0, -1, -2]])
    Z = Zonotope(c, G)
    Z_abs = abs(Z)
    
    expected_c = np.array([[1], [2], [3]])
    expected_G = np.array([[1, 2, 3], [1, 0, 2], [0, 1, 2]])
    assert np.allclose(Z_abs.c, expected_c)
    assert np.allclose(Z_abs.G, expected_G)
    
    # Test zonotope with only center (no generators)
    c = np.array([[-5], [3]])
    Z = Zonotope(c)
    Z_abs = abs(Z)
    expected_c = np.array([[5], [3]])
    assert np.allclose(Z_abs.c, expected_c)
    assert Z_abs.G.shape[1] == 0
    
    # Test properties preservation
    Z = Zonotope(np.array([[-1], [2]]), np.array([[2, -1], [-3, 1]]))
    Z_abs = abs(Z)
    
    # Dimension should be preserved
    assert Z_abs.dim() == Z.dim()
    
    # Number of generators should be preserved  
    assert Z_abs.G.shape[1] == Z.G.shape[1]
    
    # All entries should be non-negative
    assert np.all(Z_abs.c >= 0)
    assert np.all(Z_abs.G >= 0)


if __name__ == "__main__":
    test_zonotope_abs()
    print("All zonotope abs tests passed!") 