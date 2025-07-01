"""
Test file for interval power operation

Authors: Python AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.interval.power import power
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalPower:
    
    def test_power_positive_base_integer_exponent(self):
        """Test power with positive base and integer exponent"""
        I = Interval(np.array([[2]]), np.array([[3]]))
        result = I.power(2)
        
        assert np.allclose(result.inf, [[4]])
        assert np.allclose(result.sup, [[9]])
    
    def test_power_negative_base_even_exponent(self):
        """Test power with negative base and even exponent"""
        I = Interval(np.array([[-3]]), np.array([[-2]]))
        result = I.power(2)
        
        assert np.allclose(result.inf, [[4]])
        assert np.allclose(result.sup, [[9]])
    
    def test_power_interval_crossing_zero_even_exponent(self):
        """Test power with interval crossing zero and even exponent"""
        I = Interval(np.array([[-2]]), np.array([[3]]))
        result = I.power(2)
        
        assert np.allclose(result.inf, [[0]])  # Even exponent, interval crosses zero
        assert np.allclose(result.sup, [[9]])
    
    def test_power_negative_exponent(self):
        """Test power with negative exponent"""
        I = Interval(np.array([[2]]), np.array([[4]]))
        result = I.power(-1)
        
        assert np.allclose(result.inf, [[0.25]])
        assert np.allclose(result.sup, [[0.5]])
    
    def test_power_real_exponent_positive_base(self):
        """Test power with real exponent and positive base"""
        I = Interval(np.array([[1]]), np.array([[4]]))
        result = I.power(0.5)
        
        assert np.allclose(result.inf, [[1]])
        assert np.allclose(result.sup, [[2]])
    
    def test_power_real_exponent_negative_base_error(self):
        """Test power with real exponent and negative base should raise error"""
        I = Interval(np.array([[-2]]), np.array([[-1]]))
        
        with pytest.raises(CORAerror):
            I.power(0.5)
    
    def test_power_scalar_base_matrix_exponent(self):
        """Test power with scalar base and matrix exponent"""
        I = Interval(np.array([[2]]), np.array([[3]]))
        exponent = np.array([[2, 3], [1, 4]])
        result = I.power(exponent)
        
        expected_inf = np.array([[4, 8], [2, 16]])
        expected_sup = np.array([[9, 27], [3, 81]])
        
        assert np.allclose(result.inf, expected_inf)
        assert np.allclose(result.sup, expected_sup)
    
    def test_power_matrix_base_matrix_exponent(self):
        """Test power with matrix base and matrix exponent"""
        I = Interval(np.array([[1, 2], [2, 3]]), np.array([[2, 3], [3, 4]]))
        exponent = np.array([[2, 2], [2, 2]])
        result = I.power(exponent)
        
        expected_inf = np.array([[1, 4], [4, 9]])
        expected_sup = np.array([[4, 9], [9, 16]])
        
        assert np.allclose(result.inf, expected_inf)
        assert np.allclose(result.sup, expected_sup)
    
    def test_power_numeric_base_interval_exponent(self):
        """Test power with numeric base and interval exponent"""
        base = 2
        I = Interval(np.array([[1]]), np.array([[3]]))
        result = I.power(base)  # Note: This is actually base ** I
        
        # This tests the case where exponent is interval and base is numeric
        exponent_I = Interval(np.array([[1]]), np.array([[3]]))
        result = power(2, exponent_I)
        
        assert np.allclose(result.inf, [[2]])
        assert np.allclose(result.sup, [[8]])
    
    def test_power_interval_base_interval_exponent(self):
        """Test power with interval base and interval exponent"""
        base_I = Interval(np.array([[2]]), np.array([[3]]))
        exp_I = Interval(np.array([[1]]), np.array([[2]]))
        result = power(base_I, exp_I)
        
        # All combinations: 2^1, 2^2, 3^1, 3^2
        # Min: 2, Max: 9
        assert np.allclose(result.inf, [[2]])
        assert np.allclose(result.sup, [[9]])
    
    def test_power_operator_overloading(self):
        """Test power using ** operator"""
        I = Interval(np.array([[2]]), np.array([[3]]))
        result = I ** 2
        
        assert np.allclose(result.inf, [[4]])
        assert np.allclose(result.sup, [[9]])
    
    def test_power_zero_exponent(self):
        """Test power with zero exponent"""
        I = Interval(np.array([[2]]), np.array([[3]]))
        result = I.power(0)
        
        assert np.allclose(result.inf, [[1]])
        assert np.allclose(result.sup, [[1]])
    
    def test_power_complex_interval(self):
        """Test power with complex interval scenarios"""
        # Test interval that contains zero with odd exponent
        I = Interval(np.array([[-2]]), np.array([[3]]))
        result = I.power(3)
        
        assert np.allclose(result.inf, [[-8]])  # min(-2^3, 3^3)
        assert np.allclose(result.sup, [[27]])  # max(-2^3, 3^3) 