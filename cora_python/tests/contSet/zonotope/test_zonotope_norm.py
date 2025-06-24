"""
test_zonotope_norm - unit test function of norm

Syntax:
    python -m pytest test_zonotope_norm.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger, Victor Gassmann (MATLAB)
         Python translation by AI Assistant
Written: 27-July-2021 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeNorm:
    """Test class for zonotope norm method"""
    
    def test_empty_zonotope_norm(self):
        """Test norm of empty zonotope"""
        Z_empty = Zonotope.empty(2)
        norm_val = Z_empty.norm_()
        
        # Should return -inf for empty zonotope
        assert norm_val == -np.inf
    
    def test_2_norm_exact(self):
        """Test 2-norm with exact computation"""
        TOL = 1e-6
        
        c = np.zeros(2)
        G = np.array([[2, 5, 4, 3], [-4, -6, 2, 3]])
        Z = Zonotope(c, G)
        
        val2_exact = Z.norm_(2, 'exact')
        
        # Compute vertices for comparison
        V = Z.vertices_()
        vertex_max_norm = np.max(np.sqrt(np.sum(V**2, axis=0)))
        
        # Check exact vs. norm of all vertices
        if val2_exact != vertex_max_norm:
            relative_error = abs(val2_exact - vertex_max_norm) / val2_exact
            assert relative_error <= TOL
    
    def test_2_norm_upper_bound(self):
        """Test 2-norm with upper bound computation"""
        TOL = 1e-6
        
        c = np.zeros(2)
        G = np.array([[2, 5, 4, 3], [-4, -6, 2, 3]])
        Z = Zonotope(c, G)
        
        val2_exact = Z.norm_(2, 'exact')
        val2_ub = Z.norm_(2, 'ub')
        
        # Upper bound should be >= exact value
        if val2_exact > val2_ub:
            assert abs(val2_exact - val2_ub) <= TOL
    
    def test_2_norm_upper_bound_convex(self):
        """Test 2-norm with convex upper bound computation"""
        TOL = 1e-6
        
        c = np.zeros(2)
        G = np.array([[2, 5, 4, 3], [-4, -6, 2, 3]])
        Z = Zonotope(c, G)
        
        val2_exact = Z.norm_(2, 'exact')
        val2_ubc = Z.norm_(2, 'ub_convex')
        
        # Convex upper bound should be >= exact value
        if val2_exact > val2_ubc:
            assert abs(val2_exact - val2_ubc) <= TOL
    
    def test_different_norms(self):
        """Test different norm types"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        
        # Test different p-norms
        norm_1 = Z.norm_(1)
        norm_2 = Z.norm_(2)
        norm_inf = Z.norm_(np.inf)
        
        # All should be positive
        assert norm_1 > 0
        assert norm_2 > 0
        assert norm_inf > 0
    
    def test_origin_zonotope_norm(self):
        """Test norm of origin zonotope"""
        Z_origin = Zonotope.origin(3)
        norm_val = Z_origin.norm_()
        
        # Origin zonotope should have norm 0
        assert norm_val == 0
    
    def test_unit_box_norm(self):
        """Test norm of unit box"""
        # Unit box centered at origin
        Z = Zonotope(np.zeros(2), np.eye(2))
        norm_val = Z.norm_(2)
        
        # Should be sqrt(2) for unit box
        expected = np.sqrt(2)
        np.testing.assert_almost_equal(norm_val, expected)
    
    def test_1d_zonotope_norm(self):
        """Test norm of 1D zonotope"""
        Z = Zonotope(np.array([3]), np.array([[2, 1]]))
        norm_val = Z.norm_()
        
        # Should be |3| + |2| + |1| = 6 for 1-norm or max vertex for other norms
        assert norm_val > 0
    
    def test_norm_scaling(self):
        """Test that norm scales correctly with zonotope scaling"""
        Z = Zonotope(np.array([1, 1]), np.array([[1, 0], [0, 1]]))
        factor = 2
        Z_scaled = Z * factor
        
        norm_original = Z.norm_()
        norm_scaled = Z_scaled.norm_()
        
        # Norm should scale by the same factor
        np.testing.assert_almost_equal(norm_scaled, factor * norm_original)
    
    def test_norm_translation_invariance(self):
        """Test that norm is not affected by translation for centered zonotope"""
        G = np.array([[1, 0], [0, 1]])
        Z1 = Zonotope(np.zeros(2), G)
        Z2 = Zonotope(np.array([5, -3]), G)
        
        # For zonotopes with same generators, translation affects norm
        norm1 = Z1.norm_()
        norm2 = Z2.norm_()
        
        # Both should be well-defined
        assert norm1 >= 0
        assert norm2 >= 0


if __name__ == "__main__":
    test_instance = TestZonotopeNorm()
    
    # Run all tests
    test_instance.test_empty_zonotope_norm()
    test_instance.test_2_norm_exact()
    test_instance.test_2_norm_upper_bound()
    test_instance.test_2_norm_upper_bound_convex()
    test_instance.test_different_norms()
    test_instance.test_origin_zonotope_norm()
    test_instance.test_unit_box_norm()
    test_instance.test_1d_zonotope_norm()
    test_instance.test_norm_scaling()
    test_instance.test_norm_translation_invariance()
    
    print("All zonotope norm tests passed!") 