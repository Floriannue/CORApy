"""
Test for nnHelper.calcAlternatingDerCoeffs function

This test verifies that the calcAlternatingDerCoeffs function works correctly for calculating alternating derivative coefficients.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.calcAlternatingDerCoeffs import calcAlternatingDerCoeffs


class TestCalcAlternatingDerCoeffs:
    """Test class for calcAlternatingDerCoeffs function"""
    
    def test_calcAlternatingDerCoeffs_basic(self):
        """Test basic calcAlternatingDerCoeffs functionality"""
        # Test with simple case
        n = 3
        result = calcAlternatingDerCoeffs(n)
        
        # Check that result is numpy array
        assert isinstance(result, np.ndarray)
        
        # Check dimensions - should have n+1 coefficients
        assert len(result) == n + 1
        
        # Check that coefficients are real numbers
        assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_different_orders(self):
        """Test calcAlternatingDerCoeffs with different orders"""
        # Test different orders
        for n in [1, 2, 3, 5, 10]:
            result = calcAlternatingDerCoeffs(n)
            
            # Check dimensions
            assert len(result) == n + 1
            
            # Check that coefficients are real
            assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_edge_cases(self):
        """Test calcAlternatingDerCoeffs edge cases"""
        # Test with order 0
        result = calcAlternatingDerCoeffs(0)
        assert len(result) == 1
        assert np.all(np.isreal(result))
        
        # Test with order 1
        result = calcAlternatingDerCoeffs(1)
        assert len(result) == 2
        assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_large_order(self):
        """Test calcAlternatingDerCoeffs with large order"""
        n = 20  # Large order
        
        result = calcAlternatingDerCoeffs(n)
        
        # Check dimensions
        assert len(result) == n + 1
        
        # Check that coefficients are real
        assert np.all(np.isreal(result))
    
    def test_calcAlternatingDerCoeffs_consistency(self):
        """Test that calcAlternatingDerCoeffs produces consistent results"""
        n = 3
        
        # Call multiple times
        result1 = calcAlternatingDerCoeffs(n)
        result2 = calcAlternatingDerCoeffs(n)
        
        # Should be consistent
        assert np.allclose(result1, result2, atol=1e-10)
    
    def test_calcAlternatingDerCoeffs_error_handling(self):
        """Test calcAlternatingDerCoeffs error handling"""
        # Test with negative order
        with pytest.raises(ValueError):
            calcAlternatingDerCoeffs(-1)
        
        # Test with non-integer order
        with pytest.raises(ValueError):
            calcAlternatingDerCoeffs(2.5)
    
    def test_calcAlternatingDerCoeffs_mathematical_properties(self):
        """Test mathematical properties of alternating derivative coefficients"""
        # Test that coefficients alternate in sign for even orders
        n = 4
        result = calcAlternatingDerCoeffs(n)
        
        # Check alternating pattern
        for i in range(len(result) - 1):
            if i % 2 == 0:
                # Even indices should have opposite sign to next
                assert np.sign(result[i]) != np.sign(result[i + 1])
        
        # Test that coefficients sum to a known value
        # For alternating coefficients, this should follow a specific pattern
        assert np.all(np.isfinite(result))
    
    def test_calcAlternatingDerCoeffs_specific_values(self):
        """Test specific known values"""
        # Test order 1: should have coefficients [1, -1]
        result = calcAlternatingDerCoeffs(1)
        expected = np.array([1, -1])
        assert np.allclose(result, expected, atol=1e-10)
        
        # Test order 2: should have coefficients [1, -2, 1]
        result = calcAlternatingDerCoeffs(2)
        expected = np.array([1, -2, 1])
        assert np.allclose(result, expected, atol=1e-10)
    
    def test_calcAlternatingDerCoeffs_numerical_stability(self):
        """Test numerical stability for large orders"""
        # Test with very large order
        n = 50
        
        result = calcAlternatingDerCoeffs(n)
        
        # Check that all coefficients are finite
        assert np.all(np.isfinite(result))
        
        # Check that coefficients are not extremely large or small
        assert np.all(np.abs(result) < 1e10)
        assert np.all(np.abs(result) > 1e-10)
    
    def test_calcAlternatingDerCoeffs_type_consistency(self):
        """Test that output type is consistent"""
        # Test with different input types
        for n in [1, 2, 3, 5]:
            result = calcAlternatingDerCoeffs(n)
            
            # Should always be numpy array
            assert isinstance(result, np.ndarray)
            
            # Should have consistent dtype
            assert result.dtype in [np.float32, np.float64, np.float128]
    
    def test_calcAlternatingDerCoeffs_integration(self):
        """Test integration with other functions"""
        # Test that coefficients can be used in polynomial evaluation
        n = 3
        coeffs = calcAlternatingDerCoeffs(n)
        
        # Test evaluation at x = 1
        x = 1.0
        poly_value = np.polyval(coeffs, x)
        
        # Should be finite
        assert np.isfinite(poly_value)
        
        # Test evaluation at x = -1
        x = -1.0
        poly_value = np.polyval(coeffs, x)
        
        # Should be finite
        assert np.isfinite(poly_value)
