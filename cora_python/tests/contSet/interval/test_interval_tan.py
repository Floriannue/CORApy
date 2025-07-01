"""
Test file for interval tan operation

Authors: Python AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalTan:
    
    def test_tan_small_interval(self):
        """Test tan with small interval that doesn't cross discontinuities"""
        I = Interval(np.array([[0.1]]), np.array([[0.4]]))
        result = I.tan()
        
        # For small intervals, tan should be monotonic
        expected_inf = np.tan(0.1)
        expected_sup = np.tan(0.4)
        
        assert np.allclose(result.inf, [[expected_inf]])
        assert np.allclose(result.sup, [[expected_sup]])
    
    def test_tan_wide_interval(self):
        """Test tan with interval wider than pi"""
        I = Interval(np.array([[0]]), np.array([[4]]))  # Width > pi
        result = I.tan()
        
        # Should return [-inf, inf] for wide intervals
        assert np.isinf(result.inf) and result.inf < 0
        assert np.isinf(result.sup) and result.sup > 0
    
    def test_tan_interval_crossing_asymptote(self):
        """Test tan with interval crossing pi/2 asymptote"""
        I = Interval(np.array([[1.4]]), np.array([[1.8]]))  # Crosses pi/2 ≈ 1.57
        result = I.tan()
        
        # Should return [-inf, inf] when crossing asymptote
        assert np.isinf(result.inf) and result.inf < 0
        assert np.isinf(result.sup) and result.sup > 0
    
    def test_tan_negative_interval(self):
        """Test tan with negative interval"""
        I = Interval(np.array([[-0.4]]), np.array([[-0.1]]))
        result = I.tan()
        
        # For small negative intervals, tan should be monotonic
        expected_inf = np.tan(-0.4)
        expected_sup = np.tan(-0.1)
        
        assert np.allclose(result.inf, [[expected_inf]])
        assert np.allclose(result.sup, [[expected_sup]])
    
    def test_tan_interval_near_zero(self):
        """Test tan with interval near zero"""
        I = Interval(np.array([[-0.1]]), np.array([[0.1]]))
        result = I.tan()
        
        # Should be monotonic near zero
        expected_inf = np.tan(-0.1)
        expected_sup = np.tan(0.1)
        
        assert np.allclose(result.inf, [[expected_inf]])
        assert np.allclose(result.sup, [[expected_sup]])
    
    def test_tan_matrix_interval(self):
        """Test tan with matrix interval"""
        I = Interval(np.array([[0.1, 0.2], [0.3, 0.0]]), 
                    np.array([[0.2, 0.3], [0.4, 4.0]]))  # Last element: [0, 4] has width > π
        result = I.tan()
        
        # First three elements should be finite (small intervals)
        assert np.isfinite(result.inf[0, 0])
        assert np.isfinite(result.sup[0, 0])
        assert np.isfinite(result.inf[0, 1])
        assert np.isfinite(result.sup[0, 1])
        assert np.isfinite(result.inf[1, 0])
        assert np.isfinite(result.sup[1, 0])
        
        # Last element should be infinite (wide interval with width > π)
        assert np.isinf(result.inf[1, 1]) and result.inf[1, 1] < 0
        assert np.isinf(result.sup[1, 1]) and result.sup[1, 1] > 0
    
    def test_tan_boundary_case(self):
        """Test tan with interval exactly at pi width"""
        I = Interval(np.array([[0]]), np.array([[np.pi]]))
        result = I.tan()
        
        # Should return [-inf, inf] for interval of width pi
        assert np.isinf(result.inf) and result.inf < 0
        assert np.isinf(result.sup) and result.sup > 0
    
    def test_tan_discontinuity_detection(self):
        """Test tan discontinuity detection"""
        # Interval that crosses discontinuity but is narrower than pi
        I = Interval(np.array([[1.5]]), np.array([[1.6]]))  # Around pi/2 ≈ 1.57
        result = I.tan()
        
        # Check if tan(inf) > tan(sup) indicates discontinuity
        tan_inf = np.tan(1.5)
        tan_sup = np.tan(1.6)
        
        if tan_inf > tan_sup:  # Discontinuity detected
            assert np.isinf(result.inf) and result.inf < 0
            assert np.isinf(result.sup) and result.sup > 0
        else:  # No discontinuity
            assert np.allclose(result.inf, [[tan_inf]])
            assert np.allclose(result.sup, [[tan_sup]]) 