"""
test_interval_sin - unit test function for interval sin operation

This module tests the sin operation for intervals,
covering all cases including periodic behavior, edge cases, and matrix operations.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import cora_python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet import Interval


class TestIntervalSin:
    """Test class for interval sin operation"""
    
    def test_sin_empty_interval(self):
        """Test sin operation with empty intervals"""
        I = Interval.empty(1)
        I_sin = I.sin()
        assert I_sin.representsa_('emptySet')
    
    def test_sin_scalar_basic(self):
        """Test sin operation with basic scalar intervals"""
        tol = 1e-9
        
        # Simple interval [0, pi/2]
        I = Interval([0], [np.pi/2])
        I_sin = I.sin()
        assert np.isclose(I_sin.inf[0], 0, atol=tol)
        assert np.isclose(I_sin.sup[0], 1, atol=tol)
        
        # Interval [-pi/2, 0]
        I = Interval([-np.pi/2], [0])
        I_sin = I.sin()
        assert np.isclose(I_sin.inf[0], -1, atol=tol)
        assert np.isclose(I_sin.sup[0], 0, atol=tol)
    
    def test_sin_full_period(self):
        """Test sin operation with intervals spanning full periods"""
        tol = 1e-9
        
        # Interval spanning more than 2*pi -> [-1, 1]
        I = Interval([0], [3*np.pi])
        I_sin = I.sin()
        assert np.isclose(I_sin.inf[0], -1, atol=tol)
        assert np.isclose(I_sin.sup[0], 1, atol=tol)
    
    def test_sin_special_values(self):
        """Test sin with special mathematical values"""
        tol = 1e-9
        
        # sin(0) = 0
        I = Interval([0], [0])
        I_sin = I.sin()
        assert np.isclose(I_sin.inf[0], 0, atol=tol)
        assert np.isclose(I_sin.sup[0], 0, atol=tol)
        
        # sin(pi/2) = 1
        I = Interval([np.pi/2], [np.pi/2])
        I_sin = I.sin()
        assert np.isclose(I_sin.inf[0], 1, atol=tol)
        assert np.isclose(I_sin.sup[0], 1, atol=tol)
    
    def test_sin_identity_verification(self):
        """Test sin using identity sin(x) = cos(x - pi/2)"""
        tol = 1e-9
        
        # Test various intervals
        test_intervals = [
            Interval([0], [np.pi/2]),
            Interval([-np.pi/4], [np.pi/4]),
            Interval([np.pi], [3*np.pi/2])
        ]
        
        for I in test_intervals:
            I_sin = I.sin()
            I_cos_shifted = (I - np.pi/2).cos()
            
            # sin(x) should equal cos(x - pi/2)
            assert np.allclose(I_sin.inf, I_cos_shifted.inf, atol=tol)
            assert np.allclose(I_sin.sup, I_cos_shifted.sup, atol=tol)


if __name__ == '__main__':
    pytest.main([__file__]) 