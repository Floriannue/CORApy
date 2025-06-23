"""
test_zonotope_zonotopeNorm - unit test function of zonotope norm

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-January-2024 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.zonotope.zonotopeNorm import zonotopeNorm


class TestZonotopeZonotopeNorm:
    """Test class for zonotope zonotopeNorm function"""
    
    def test_empty_zonotope(self):
        """Test zonotopeNorm with empty zonotopes"""
        # Empty zonotope
        Z = Zonotope.empty(2)
        
        # Point in non-empty space should give infinite norm
        res, minimizer = zonotopeNorm(Z, np.array([1, -1]))
        assert res == np.inf
        assert minimizer.size == 0
        
        # Empty point with empty zonotope should give 0 norm
        res, minimizer = zonotopeNorm(Z, np.empty((2, 0)))
        assert res == 0
        assert minimizer.size == 0
    
    def test_point_zonotope(self):
        """Test zonotopeNorm with zonotope that is only a center point"""
        # 2D, only center
        Z = Zonotope(np.array([0, 0]))
        
        # Zero point should give norm 0
        res, minimizer = zonotopeNorm(Z, np.array([0, 0]))
        assert res == 0
        assert minimizer.size == 0
        
        # Non-zero point should give infinite norm
        res, minimizer = zonotopeNorm(Z, np.array([1, 0]))
        assert res == np.inf
        assert minimizer.size == 0
    
    def test_standard_zonotope(self):
        """Test zonotopeNorm with standard zonotope cases"""
        # 2D, center and generators
        c = np.array([0, 0])
        G = np.array([[1, -2, 2, 0], 
                      [-1, 1, 0, 1]])
        Z = Zonotope(c, G)
        
        # Test specific points from MATLAB test
        res, minimizer = zonotopeNorm(Z, np.array([5, 3]))
        assert abs(res - 2.2) < 1e-10, f"Expected 2.2, got {res}"
        
        res, minimizer = zonotopeNorm(Z, np.array([-5, 3]))
        assert abs(res - 1.0) < 1e-10, f"Expected 1.0, got {res}"
        
        # Verify that minimizer actually satisfies the constraint G * minimizer = p
        if minimizer.size > 0:
            p_reconstructed = G @ minimizer.reshape(-1, 1)
            np.testing.assert_allclose(p_reconstructed.flatten(), [-5, 3], atol=1e-10)
    
    def test_shifted_center(self):
        """Test that shifted center does not influence the result"""
        # Original zonotope
        c = np.array([0, 0])
        G = np.array([[1, -2, 2, 0], 
                      [-1, 1, 0, 1]])
        Z = Zonotope(c, G)
        
        # Shifted zonotope
        Z_shifted = Z + np.array([1, -2])
        
        # Both should give same results
        res1, _ = zonotopeNorm(Z, np.array([5, 3]))
        res2, _ = zonotopeNorm(Z_shifted, np.array([5, 3]))
        assert abs(res1 - res2) < 1e-10
        
        res1, _ = zonotopeNorm(Z, np.array([-5, 3]))
        res2, _ = zonotopeNorm(Z_shifted, np.array([-5, 3]))
        assert abs(res1 - res2) < 1e-10
    
    def test_input_validation(self):
        """Test input validation"""
        # Valid zonotope
        Z = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
        
        # Invalid zonotope type
        with pytest.raises(ValueError, match="First argument must be a zonotope"):
            zonotopeNorm("not a zonotope", np.array([1, 2]))
        
        # Invalid point type
        with pytest.raises(ValueError, match="Second argument must be a numeric array"):
            zonotopeNorm(Z, "not an array")
        
        # Dimension mismatch should be caught by equal_dim_check
        with pytest.raises(Exception):  # equal_dim_check may raise different exception types
            zonotopeNorm(Z, np.array([1, 2, 3]))  # 3D point for 2D zonotope
    
    def test_norm_properties(self):
        """Test mathematical properties of the zonotope norm"""
        # Create a simple zonotope
        c = np.array([1, 1])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Test some points
        p1 = np.array([0.5, 0.5])
        p2 = np.array([1.0, 1.0])
        
        res1, min1 = zonotopeNorm(Z, p1)
        res2, min2 = zonotopeNorm(Z, p2)
        
        # Results should be finite and non-negative
        assert np.isfinite(res1) and res1 >= 0
        assert np.isfinite(res2) and res2 >= 0
        
        # Minimizers should satisfy the constraint if they exist
        if min1.size > 0:
            p1_reconstructed = G @ min1.reshape(-1, 1)
            np.testing.assert_allclose(p1_reconstructed.flatten(), p1, atol=1e-10)
        
        if min2.size > 0:
            p2_reconstructed = G @ min2.reshape(-1, 1)
            np.testing.assert_allclose(p2_reconstructed.flatten(), p2, atol=1e-10)
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # 1D zonotope
        Z_1d = Zonotope(np.array([0]), np.array([[2]]))
        res, _ = zonotopeNorm(Z_1d, np.array([1]))
        assert abs(res - 0.5) < 1e-10  # |1/2| = 0.5
        
        # High-dimensional case
        n = 5
        c = np.zeros(n)
        G = np.eye(n)
        Z_nd = Zonotope(c, G)
        
        p = np.ones(n) * 0.5
        res, min_nd = zonotopeNorm(Z_nd, p)
        assert abs(res - 0.5) < 1e-10  # All coefficients should be 0.5
        
        if min_nd.size > 0:
            np.testing.assert_allclose(min_nd, np.ones(n) * 0.5, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__]) 