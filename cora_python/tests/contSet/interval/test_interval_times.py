"""
test_interval_times - unit test function of times

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       06-April-2023
Last update:   ---
Last revision: ---
"""

import pytest
import sys
import os
import numpy as np

# Add the parent directory to the path to import cora_python modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.times import times
from cora_python.contSet.interval.representsa_ import representsa_


class TestIntervalTimes:
    
    def test_times_interval_scalar(self):
        """Test element-wise multiplication of interval with scalar"""
        # Test interval .* scalar
        I = Interval([1, 3], [2, 5])
        M = np.array([3, -1])
        MI = times(I, M)
        
        expected_inf = np.array([3, -5])
        expected_sup = np.array([6, -3])
        
        assert np.allclose(MI.inf, expected_inf), f"Expected inf {expected_inf}, got {MI.inf}"
        assert np.allclose(MI.sup, expected_sup), f"Expected sup {expected_sup}, got {MI.sup}"
    
    def test_times_scalar_Interval(self):
        """Test element-wise multiplication of scalar with interval"""
        # Test scalar .* interval
        I = Interval([1, 3], [2, 5])
        M = np.array([3, -1])
        MI = times(M, I)
        
        expected_inf = np.array([3, -5])
        expected_sup = np.array([6, -3])
        
        assert np.allclose(MI.inf, expected_inf), f"Expected inf {expected_inf}, got {MI.inf}"
        assert np.allclose(MI.sup, expected_sup), f"Expected sup {expected_sup}, got {MI.sup}"
    
    def test_times_interval_Interval(self):
        """Test element-wise multiplication of interval with interval"""
        I1 = Interval([1, -2], [2, -1])
        I2 = Interval([-1, 0], [1, 3])
        
        MI = times(I1, I2)
        
        # Expected result: element-wise multiplication
        # [1,2] .* [-1,1] = [min(1*(-1), 1*1, 2*(-1), 2*1), max(...)] = [-2, 2]
        # [-2,-1] .* [0,3] = [min(-2*0, -2*3, -1*0, -1*3), max(...)] = [-6, 0]
        expected_inf = np.array([-2, -6])
        expected_sup = np.array([2, 0])
        
        assert np.allclose(MI.inf, expected_inf), f"Expected inf {expected_inf}, got {MI.inf}"
        assert np.allclose(MI.sup, expected_sup), f"Expected sup {expected_sup}, got {MI.sup}"
    
    def test_times_empty_Interval(self):
        """Test element-wise multiplication with empty interval"""
        I = Interval.empty(2)
        result = times(2, I)
        
        # Empty interval times anything should remain empty
        assert representsa_(result, 'emptySet', 1e-12)
    
    def test_times_scalar_cases(self):
        """Test element-wise multiplication with different scalar cases"""
        I = Interval([-1, 0], [2, 3])
        
        # Positive scalar
        result1 = times(I, 2)
        assert np.allclose(result1.inf, [-2, 0])
        assert np.allclose(result1.sup, [4, 6])
        
        # Negative scalar
        result2 = times(I, -2)
        assert np.allclose(result2.inf, [-4, -6])
        assert np.allclose(result2.sup, [2, 0])
        
        # Zero scalar
        result3 = times(I, 0)
        assert np.allclose(result3.inf, [0, 0])
        assert np.allclose(result3.sup, [0, 0])
    
    def test_times_matrix_operations(self):
        """Test element-wise multiplication with matrix operations"""
        # Create 2x2 interval matrix
        inf_mat = np.array([[1, -1], [0, 2]])
        sup_mat = np.array([[2, 0], [1, 3]])
        I = Interval(inf_mat, sup_mat)
        
        # Element-wise multiply with scalar
        result = times(I, 2)
        expected_inf = 2 * inf_mat
        expected_sup = 2 * sup_mat
        
        assert np.allclose(result.inf, expected_inf)
        assert np.allclose(result.sup, expected_sup)
    
    def test_times_mixed_sign_intervals(self):
        """Test element-wise multiplication with mixed sign intervals"""
        # Interval that spans zero
        I1 = Interval([-2], [3])
        I2 = Interval([-1], [4])
        
        result = times(I1, I2)
        
        # All possible products: (-2)*(-1)=2, (-2)*4=-8, 3*(-1)=-3, 3*4=12
        # So result should be [-8, 12]
        assert np.allclose(result.inf, [-8])
        assert np.allclose(result.sup, [12])
    
    def test_times_zero_containing_intervals(self):
        """Test element-wise multiplication with intervals containing zero"""
        I1 = Interval([-1], [1])  # Contains zero
        I2 = Interval([2], [3])   # Positive
        
        result = times(I1, I2)
        
        # Products: (-1)*2=-2, (-1)*3=-3, 1*2=2, 1*3=3
        # So result should be [-3, 3]
        assert np.allclose(result.inf, [-3])
        assert np.allclose(result.sup, [3])
    
    def test_times_infinite_values(self):
        """Test element-wise multiplication with infinite values"""
        I = Interval([-np.inf], [np.inf])
        
        # Multiplication with positive scalar
        result1 = times(I, 2)
        assert result1.inf[0] == -np.inf
        assert result1.sup[0] == np.inf
        
        # Multiplication with negative scalar should flip bounds
        result2 = times(I, -2)
        assert result2.inf[0] == -np.inf
        assert result2.sup[0] == np.inf
    
    def test_times_nan_handling(self):
        """Test that NaN cases are handled properly (0 * inf = 0)"""
        # This tests the 0*Inf=NaN fix mentioned in MATLAB code
        I = Interval([0], [0])  # Zero interval
        
        # Test with infinity - should result in zero interval, not NaN
        result = times(I, np.inf)
        assert np.allclose(result.inf, [0])
        assert np.allclose(result.sup, [0])


if __name__ == '__main__':
    pytest.main([__file__]) 
