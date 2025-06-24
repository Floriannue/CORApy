"""
Test file for interval vertcat method (vertical concatenation)

Authors: Matthias Althoff (MATLAB), Python translation by AI Assistant
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalVertcat:
    
    def test_vertcat_basic(self):
        """Test basic vertical concatenation"""
        I1 = Interval([-1], [1])
        I2 = Interval([1], [2])
        
        result = I1.vertcat(I2)
        
        expected_inf = np.array([[-1], [1]])
        expected_sup = np.array([[1], [2]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_vertcat_multiple(self):
        """Test concatenation of multiple intervals"""
        I1 = Interval([-1], [1])
        I2 = Interval([0], [2])
        I3 = Interval([1], [3])
        
        result = I1.vertcat(I2, I3)
        
        expected_inf = np.array([[-1], [0], [1]])
        expected_sup = np.array([[1], [2], [3]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_vertcat_with_numeric(self):
        """Test concatenation with numeric values"""
        I = Interval([-1], [1])
        numeric = 2
        
        result = I.vertcat(numeric)
        
        expected_inf = np.array([[-1], [2]])
        expected_sup = np.array([[1], [2]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_vertcat_vector_intervals(self):
        """Test concatenation of vector intervals"""
        I1 = Interval([[1, 2]], [[3, 4]])  # Row vector interval
        I2 = Interval([[5, 6]], [[7, 8]])  # Row vector interval
        
        result = I1.vertcat(I2)
        
        expected_inf = np.array([[1, 2], [5, 6]])
        expected_sup = np.array([[3, 4], [7, 8]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup)
        
    def test_vertcat_empty_args(self):
        """Test with empty argument list"""
        with pytest.raises(ValueError):
            Interval.vertcat()
            
    def test_vertcat_non_interval_first(self):
        """Test when first argument is not an interval"""
        numeric = 1
        I = Interval([2], [3])
        
        result = Interval.vertcat(numeric, I)
        
        expected_inf = np.array([[1], [2]])
        expected_sup = np.array([[1], [3]])
        
        np.testing.assert_array_equal(result.inf, expected_inf)
        np.testing.assert_array_equal(result.sup, expected_sup) 