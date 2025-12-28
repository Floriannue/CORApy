"""
test_zonotope_minus - unit test function of minus

This test file matches MATLAB's behavior exactly:
- Z - v: Allowed (translation by vector)
- v - Z: Allowed (via rminus)
- Z1 - Z2: NOT allowed (must use minkDiff)

Syntax:
    python -m pytest test_zonotope_minus.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-May-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeMinusOp:
    """Test class for zonotope minus method - matches MATLAB behavior"""
    
    def test_zonotope_minus_vector(self):
        """Test subtraction of vector from zonotope: Z - v"""
        # MATLAB: Z - v calls Z + (-v)
        Z = Zonotope(np.array([[3], [4]]), np.array([[1, 0], [0, 1]]))
        v = np.array([1, 2])
        
        Z_translated = Z - v
        
        # Should be translation by -v
        expected_center = Z.c.flatten() - v
        np.testing.assert_array_equal(Z_translated.c.flatten(), expected_center)
        
        # Generators should remain the same
        np.testing.assert_array_equal(Z_translated.G, Z.G)
    
    def test_vector_minus_zonotope(self):
        """Test subtraction of zonotope from vector: v - Z"""
        # MATLAB: v - Z calls -Z + v
        v = np.array([5, 6])
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        
        Z_result = v - Z
        
        # Should be v - Z = -Z + v
        Z_negated = -Z
        Z_expected = Z_negated + v
        
        assert Z_result.isequal(Z_expected)
    
    def test_minus_with_scalar(self):
        """Test subtraction with scalar: Z - scalar"""
        # MATLAB: scalar is treated as vector
        Z = Zonotope(np.array([[3], [4]]), np.array([[1, 0], [0, 1]]))
        scalar = 2
        
        Z_result = Z - scalar
        
        # Should subtract scalar from center (broadcast)
        expected_center = Z.c.flatten() - scalar
        np.testing.assert_array_equal(Z_result.c.flatten(), expected_center)
        
        # Generators should remain the same
        np.testing.assert_array_equal(Z_result.G, Z.G)
    
    def test_zonotope_minus_zonotope_raises_error(self):
        """Test that Z1 - Z2 raises error directing to minkDiff"""
        # MATLAB: Z1 - Z2 is NOT allowed, must use minkDiff
        Z1 = Zonotope(np.array([[2], [3]]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([[1], [1]]), np.array([[0.5, 0], [0, 0.5]]))
        
        with pytest.raises(CORAerror, match="minkDiff"):
            Z1 - Z2
    
    def test_minus_empty_zonotope(self):
        """Test subtraction with empty zonotope: Z - v where Z is empty"""
        Z_empty = Zonotope.empty(2)
        v = np.array([1, 2])
        
        # Z_empty - v should still be empty
        Z_result = Z_empty - v
        assert Z_result.isemptyobject()
    
    def test_minus_origin(self):
        """Test subtraction with origin zonotope: Z - v where Z is origin"""
        Z_origin = Zonotope.origin(2)
        v = np.array([1, 2])
        
        # Origin - v should be point at -v
        Z_result = Z_origin - v
        np.testing.assert_array_almost_equal(Z_result.c.flatten(), -v)
    
    def test_minus_1d(self):
        """Test subtraction of 1D zonotope with vector"""
        Z = Zonotope(np.array([[5]]), np.array([[2, 1]]))
        v = np.array([2])
        
        Z_result = Z - v
        
        assert Z_result.dim() == 1
        expected_center = Z.c.flatten() - v
        np.testing.assert_array_equal(Z_result.c.flatten(), expected_center)
    
    def test_minus_different_dimensions_error(self):
        """Test that subtraction with incompatible dimensions raises error"""
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        v_wrong_dim = np.array([1, 2, 3])
        
        with pytest.raises(Exception):
            Z - v_wrong_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
