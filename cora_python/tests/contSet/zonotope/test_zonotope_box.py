import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope


def test_zonotope_box():
    """
    Test box method for zonotope - computes an enclosing axis-aligned box 
    in generator representation according to manual Appendix A.1.
    """
    
    # Test empty zonotope
    Z_empty = Zonotope.empty(2)
    Z_box = Z_empty.box()
    assert Z_box.representsa_('emptySet')
    assert Z_box.dim() == 2
    
    # Test 1D zonotope
    Z = Zonotope(np.array([[2]]), np.array([[1, 0.5]]))
    Z_box = Z.box()
    expected_c = np.array([[2]])
    expected_G = np.array([[1.5]])  # sum of absolute values of generators
    assert np.allclose(Z_box.c, expected_c)
    assert np.allclose(Z_box.G, expected_G)
    
    # Test 2D zonotope - example from MATLAB test
    c = np.array([[1], [0]]) 
    G = np.array([[2, -1], [4, 1]])
    Z = Zonotope(c, G)
    Z_box = Z.box()
    
    # Expected: interval bounds are [1-3, 1+3] x [0-5, 0+5] = [-2,4] x [-5,5]
    # Center: [1, 0], radii: [3, 5]
    expected_c = np.array([[1], [0]])
    expected_G = np.array([[3, 0], [0, 5]])
    assert np.allclose(Z_box.c, expected_c)
    assert np.allclose(Z_box.G, expected_G)
    
    # Test that result is axis-aligned
    assert Z_box.representsa_('interval')
    
    # Test 2D zonotope that's already axis-aligned
    c = np.array([[0], [1]])
    G = np.array([[2, 0], [0, 3]])
    Z = Zonotope(c, G)
    Z_box = Z.box()
    
    # Should remain unchanged (up to generator ordering)
    expected_c = np.array([[0], [1]])
    expected_radii = np.array([2, 3])  # radii should match
    computed_radii = np.sort(np.abs(Z_box.G).sum(axis=1))
    expected_radii_sorted = np.sort(expected_radii)
    assert np.allclose(computed_radii, expected_radii_sorted)
    assert np.allclose(Z_box.c, expected_c)
    
    # Test 3D zonotope
    c = np.array([[1], [2], [-1]])
    G = np.array([[1, 0, 2], [0, 1, -1], [2, 1, 0]])
    Z = Zonotope(c, G)
    Z_box = Z.box()
    
    # Box should contain original zonotope
    assert Z_box.dim() == 3
    assert Z_box.representsa_('interval')
    
    # Check containment by sampling points from original zonotope
    np.random.seed(42)
    for _ in range(10):
        p = Z.randPoint()
        # Convert both to intervals for containment check
        from cora_python.contSet.interval import Interval
        I_box = Interval(Z_box)
        assert I_box.contains_(p), "Box should contain all points from original zonotope"
    
    # Test zonotope with only center (no generators)
    c = np.array([[5], [-2]])
    Z = Zonotope(c)
    Z_box = Z.box()
    
    # Should be just the point
    assert np.allclose(Z_box.c, c)
    assert Z_box.G.shape[1] == 0
    
    # Test properties preservation
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 2], [3, -1]]))
    Z_box = Z.box()
    
    # Dimension should be preserved
    assert Z_box.dim() == Z.dim()
    
    # Box should be axis-aligned
    assert Z_box.representsa_('interval')
    
    # Box should contain original zonotope (approximate check)
    from cora_python.contSet.interval import Interval
    I_orig = Interval(Z)
    I_box = Interval(Z_box)
    
    # All bounds of box should be >= bounds of original
    assert np.all(I_box.infimum() <= I_orig.infimum() + 1e-10)
    assert np.all(I_box.supremum() >= I_orig.supremum() - 1e-10)


if __name__ == "__main__":
    test_zonotope_box()
    print("All zonotope box tests passed!") 