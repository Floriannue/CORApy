"""
Test cases for interval atan method

Authors: AI Assistant
Written: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalAtan:
    """Test class for interval atan method"""
    
    def test_atan_valid_interval(self):
        """Test atan with valid intervals"""
        # Test case 1: Normal interval
        I = Interval(-1.5, 2.3)
        res = I.atan()
        
        # atan is increasing, so atan(inf) becomes inf and atan(sup) becomes sup
        expected_inf = np.arctan(-1.5)
        expected_sup = np.arctan(2.3)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_atan_boundary_values(self):
        """Test atan with large values approaching asymptotes"""
        I = Interval(-100, 100)
        res = I.atan()
        
        # atan approaches -pi/2 and pi/2 as x approaches -inf and +inf
        assert res.inf > -np.pi/2
        assert res.sup < np.pi/2
        assert res.inf < res.sup
        
    def test_atan_single_point(self):
        """Test atan with single point intervals"""
        # Test with 0
        I = Interval(0)
        res = I.atan()
        expected = np.arctan(0)  # 0
        
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
        
        # Test with 1
        I = Interval(1)
        res = I.atan()
        expected = np.arctan(1)  # pi/4
        
        assert np.allclose(res.inf, expected)
        assert np.allclose(res.sup, expected)
        
    def test_atan_matrix_intervals(self):
        """Test atan with matrix intervals"""
        # 2x2 matrix interval
        inf_mat = np.array([[-1.5, 0.2], [0.1, -2.8]])
        sup_mat = np.array([[0.3, 2.7], [1.6, -0.1]])
        
        I = Interval(inf_mat, sup_mat)
        res = I.atan()
        
        # Check that result has correct shape
        assert res.inf.shape == (2, 2)
        assert res.sup.shape == (2, 2)
        
        # Check individual elements (atan is increasing)
        expected_inf = np.arctan(inf_mat)
        expected_sup = np.arctan(sup_mat)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_atan_monotonicity(self):
        """Test that atan respects monotonicity (increasing)"""
        I1 = Interval(-2.0, -0.5)
        I2 = Interval(0.5, 2.0)
        
        res1 = I1.atan()
        res2 = I2.atan()
        
        # Since atan is increasing, larger input intervals should give larger output intervals
        assert np.all(res1.inf < res2.inf)  # atan(-2.0) < atan(0.5)
        assert np.all(res1.sup < res2.sup)  # atan(-0.5) < atan(2.0)
        
    def test_atan_vector_intervals(self):
        """Test atan with vector intervals"""
        # Vector interval
        inf_vec = np.array([-2.5, 0.1, -1.0])
        sup_vec = np.array([0.2, 3.9, 0.5])
        
        I = Interval(inf_vec, sup_vec)
        res = I.atan()
        
        # Check shape
        assert res.inf.shape == (3,)
        assert res.sup.shape == (3,)
        
        # Check values (atan is increasing)
        expected_inf = np.arctan(inf_vec)
        expected_sup = np.arctan(sup_vec)
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
    def test_atan_edge_cases(self):
        """Test atan with edge cases"""
        # Very large positive interval
        I = Interval(1000, 10000)
        res = I.atan()
        
        # Should approach pi/2 but not exceed it
        assert np.all(res.inf > 0)
        assert np.all(res.sup < np.pi/2)
        assert np.all(res.inf < res.sup)
        
        # Very large negative interval
        I = Interval(-10000, -1000)
        res = I.atan()
        
        # Should approach -pi/2 but not go below it
        assert np.all(res.inf > -np.pi/2)
        assert np.all(res.sup < 0)
        assert np.all(res.inf < res.sup)
        
    def test_atan_zero_crossing(self):
        """Test atan with intervals that cross zero"""
        I = Interval(-1.0, 1.0)
        res = I.atan()
        
        expected_inf = np.arctan(-1.0)  # -pi/4
        expected_sup = np.arctan(1.0)   # pi/4
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
        
        # Result should be symmetric around zero
        assert np.allclose(res.inf, -res.sup)
        
    def test_atan_type_consistency(self):
        """Test that atan returns proper Interval type"""
        I = Interval(-0.3, 0.7)
        res = I.atan()
        
        assert isinstance(res, Interval)
        assert hasattr(res, 'inf')
        assert hasattr(res, 'sup')
        
    def test_atan_numerical_stability(self):
        """Test numerical stability of atan implementation"""
        # Test with very small values
        I = Interval(-1e-15, 1e-15)
        res = I.atan()
        
        # Should be very close to zero
        assert np.allclose(res.inf, 0, atol=1e-14)
        assert np.allclose(res.sup, 0, atol=1e-14)
        
        # Test with very large values
        I = Interval(-1e15, 1e15)
        res = I.atan()
        
        # Should approach asymptotes
        assert res.inf > -np.pi/2
        assert res.sup < np.pi/2
        assert np.allclose(res.inf, -np.pi/2, atol=1e-10)
        assert np.allclose(res.sup, np.pi/2, atol=1e-10)
        
    def test_atan_range_bounds(self):
        """Test that atan output is always within (-pi/2, pi/2)"""
        test_intervals = [
            Interval(-1000, 1000),
            Interval(-np.inf, np.inf),
            Interval(0, np.inf),
            Interval(-np.inf, 0),
        ]
        
        for I in test_intervals:
            res = I.atan()
            
            # All results should be within the range of atan
            assert np.all(res.inf >= -np.pi/2)
            assert np.all(res.sup <= np.pi/2)
            assert np.all(res.inf <= res.sup)
            
    def test_atan_infinite_inputs(self):
        """Test atan with infinite inputs"""
        # Positive infinity
        I = Interval(0, np.inf)
        res = I.atan()
        
        assert np.allclose(res.inf, 0)
        assert np.allclose(res.sup, np.pi/2, atol=1e-10)
        
        # Negative infinity
        I = Interval(-np.inf, 0)
        res = I.atan()
        
        assert np.allclose(res.inf, -np.pi/2, atol=1e-10)
        assert np.allclose(res.sup, 0)
        
        # Both infinities
        I = Interval(-np.inf, np.inf)
        res = I.atan()
        
        assert np.allclose(res.inf, -np.pi/2, atol=1e-10)
        assert np.allclose(res.sup, np.pi/2, atol=1e-10) 