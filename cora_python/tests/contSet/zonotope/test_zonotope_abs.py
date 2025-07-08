"""
test_zonotope_abs - unit test function of abs

Syntax:
    python -m pytest test_zonotope_abs.py

Inputs:
    -

Outputs:
    test results

Authors: Python translation by AI Assistant
Written: 2025
"""

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


class TestZonotopeAbs:
    """Test class for zonotope abs method"""
    
    def test_positive_zonotope_abs(self):
        """Test absolute value of zonotope with all positive values"""
        # Zonotope entirely in positive quadrant
        c = np.array([3, 4])
        G = np.array([[1, 0.5], [0, 1]])
        Z = Zonotope(c, G)
        
        Z_abs = Z.abs()
        
        # Should be equal to original since all values are positive
        assert Z.isequal(Z_abs)
    
    def test_negative_zonotope_abs(self):
        """Test absolute value of zonotope with all negative values"""
        # Zonotope entirely in negative quadrant
        c = np.array([-3, -4])
        G = np.array([[-1, -0.5], [0, -1]])
        Z = Zonotope(c, G)
        
        Z_abs = Z.abs()
        
        # All coordinates should be made positive
        # Check by sampling points
        points = Z_abs.randPoint_(20)
        for i in range(points.shape[1]):
            assert np.all(points[:, i] >= 0)
    
    def test_mixed_sign_zonotope_abs(self):
        """Test absolute value of zonotope spanning positive and negative regions"""
        # Zonotope centered at origin
        c = np.array([0, 0])
        G = np.array([[2, 0], [0, 3]])
        Z = Zonotope(c, G)
        
        Z_abs = Z.abs()
        
        # Result should be in positive quadrant
        points = Z_abs.randPoint_(50)
        for i in range(points.shape[1]):
            assert np.all(points[:, i] >= 0)
    
    def test_1d_abs(self):
        """Test absolute value of 1D zonotope"""
        # 1D zonotope spanning from -5 to 3
        c = np.array([-1])
        G = np.array([[4]])  # Spans from -5 to 3
        Z = Zonotope(c, G)
        
        Z_abs = Z.abs()
        
        # Result should span from 0 to 5
        points = Z_abs.randPoint_(30)
        assert np.all(points >= 0)
        assert np.max(points) <= 5.1  # Allow small numerical tolerance
    
    def test_origin_abs(self):
        """Test absolute value of origin zonotope"""
        Z_origin = Zonotope.origin(3)
        Z_abs = Z_origin.abs()
        
        # Should remain origin
        assert Z_origin.isequal(Z_abs)
    
    def test_empty_zonotope_abs(self):
        """Test absolute value of empty zonotope"""
        Z_empty = Zonotope.empty(2)
        Z_abs = Z_empty.abs()
        
        assert Z_abs.isemptyobject()
        assert Z_abs.dim() == 2
    
    def test_abs_containment(self):
        """Test that abs of individual points is contained in abs of zonotope"""
        c = np.array([1, -2])
        G = np.array([[2, 1], [1, 2]])
        Z = Zonotope(c, G)
        
        Z_abs = Z.abs()
        
        # Sample points from original zonotope
        points = Z.randPoint_(30)
        
        # Take absolute value of each point
        points_abs = np.abs(points)
        
        # Each absolute point should be contained in absolute zonotope
        for i in range(points.shape[1]):
            assert Z_abs.contains_(points_abs[:, i])
    
    def test_abs_idempotent(self):
        """Test that abs(abs(Z)) = abs(Z)"""
        c = np.array([-1, 2])
        G = np.array([[3, 1], [-2, 1]])
        Z = Zonotope(c, G)
        
        Z_abs = Z.abs()
        Z_abs_abs = Z_abs.abs()
        
        # Taking absolute value twice should be the same as once
        assert Z_abs.isequal(Z_abs_abs)
    
    def test_abs_positive_orthant(self):
        """Test that result is always in positive orthant"""
        # Test with random zonotopes
        for _ in range(5):
            c = np.random.randn(3) * 5  # Random center
            G = np.random.randn(3, 4) * 2  # Random generators
            Z = Zonotope(c, G)
            
            Z_abs = Z.abs()
            
            # Sample points and verify all are non-negative
            points = Z_abs.randPoint_(20)
            for i in range(points.shape[1]):
                assert np.all(points[:, i] >= -1e-10)  # Allow small numerical errors
    
    def test_abs_dimension_preservation(self):
        """Test that absolute value preserves dimension"""
        for dim in [1, 2, 3, 5]:
            c = np.random.randn(dim)
            G = np.random.randn(dim, dim + 2)
            Z = Zonotope(c, G)
            
            Z_abs = Z.abs()
            
            assert Z_abs.dim() == dim


if __name__ == "__main__":
    test_instance = TestZonotopeAbs()
    
    # Run all tests
    test_instance.test_positive_zonotope_abs()
    test_instance.test_negative_zonotope_abs()
    test_instance.test_mixed_sign_zonotope_abs()
    test_instance.test_1d_abs()
    test_instance.test_origin_abs()
    test_instance.test_empty_zonotope_abs()
    test_instance.test_abs_containment()
    test_instance.test_abs_idempotent()
    test_instance.test_abs_positive_orthant()
    test_instance.test_abs_dimension_preservation()
    
    print("All zonotope abs tests passed!") 