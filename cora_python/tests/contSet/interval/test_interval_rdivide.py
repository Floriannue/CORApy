"""
Test file for interval rdivide operation

Authors: Python AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalRdivide:
    
    def test_rdivide_interval_by_scalar(self):
        """Test interval divided by scalar"""
        I = Interval(np.array([[2]]), np.array([[4]]))
        result = I.rdivide(2)
        
        assert np.allclose(result.inf, [[1]])
        assert np.allclose(result.sup, [[2]])
    
    def test_rdivide_interval_by_negative_scalar(self):
        """Test interval divided by negative scalar"""
        I = Interval(np.array([[2]]), np.array([[4]]))
        result = I.rdivide(-2)
        
        assert np.allclose(result.inf, [[-2]])
        assert np.allclose(result.sup, [[-1]])
    
    def test_rdivide_interval_by_zero(self):
        """Test interval divided by zero gives infinite interval"""
        I = Interval(np.array([[2]]), np.array([[4]]))
        result = I.rdivide(0)
        
        assert np.isinf(result.inf) and result.inf < 0
        assert np.isinf(result.sup) and result.sup > 0
    
    def test_rdivide_scalar_by_interval(self):
        """Test scalar divided by interval"""
        I = Interval(np.array([[2]]), np.array([[4]]))
        result = I.rdivide(8)  # This is actually rdivide(8, I)
        
        # Use the function directly for scalar/interval
        from cora_python.contSet.interval.rdivide import rdivide
        result = rdivide(8, I)
        
        assert np.allclose(result.inf, [[2]])
        assert np.allclose(result.sup, [[4]])
    
    def test_rdivide_scalar_by_interval_crossing_zero(self):
        """Test scalar divided by interval that crosses zero"""
        I = Interval(np.array([[-2]]), np.array([[3]]))
        from cora_python.contSet.interval.rdivide import rdivide
        result = rdivide(6, I)
        
        # Should give [-inf, inf] since interval crosses zero
        assert np.isinf(result.inf) and result.inf < 0
        assert np.isinf(result.sup) and result.sup > 0
    
    def test_rdivide_scalar_by_interval_inf_zero(self):
        """Test scalar divided by interval with inf = 0"""
        I = Interval(np.array([[0]]), np.array([[2]]))
        from cora_python.contSet.interval.rdivide import rdivide
        result = rdivide(4, I)
        
        # Should give [2, inf]
        assert np.allclose(result.inf, [[2]])
        assert np.isinf(result.sup) and result.sup > 0
    
    def test_rdivide_scalar_by_interval_sup_zero(self):
        """Test scalar divided by interval with sup = 0"""
        I = Interval(np.array([[-2]]), np.array([[0]]))
        from cora_python.contSet.interval.rdivide import rdivide
        result = rdivide(4, I)
        
        # Should give [-inf, -2]
        assert np.isinf(result.inf) and result.inf < 0
        assert np.allclose(result.sup, [[-2]])
    
    def test_rdivide_scalar_by_zero_interval_error(self):
        """Test scalar divided by zero interval should give NaN"""
        I = Interval(np.array([[0]]), np.array([[0]]))
        from cora_python.contSet.interval.rdivide import rdivide
        
        with pytest.raises(CORAerror):
            rdivide(4, I)
    
    def test_rdivide_interval_by_interval(self):
        """Test interval divided by interval"""
        I1 = Interval(np.array([[4]]), np.array([[8]]))
        I2 = Interval(np.array([[2]]), np.array([[4]]))
        result = I1.rdivide(I2)  # This uses I1 / I2 = I1 * (1/I2)
        
        # Expected: [4/4, 8/2] = [1, 4]
        assert np.allclose(result.inf, [[1]])
        assert np.allclose(result.sup, [[4]])
    
    def test_rdivide_matrix_intervals(self):
        """Test rdivide with matrix intervals"""
        I = Interval(np.array([[2, 4], [6, 8]]), np.array([[4, 6], [8, 10]]))
        divisor = np.array([[2, 2], [2, 2]])
        result = I.rdivide(divisor)
        
        expected_inf = np.array([[1, 2], [3, 4]])
        expected_sup = np.array([[2, 3], [4, 5]])
        
        assert np.allclose(result.inf, expected_inf)
        assert np.allclose(result.sup, expected_sup)
    
    def test_rdivide_operator_overloading(self):
        """Test rdivide using / operator"""
        I = Interval(np.array([[4]]), np.array([[8]]))
        result = I / 2
        
        assert np.allclose(result.inf, [[2]])
        assert np.allclose(result.sup, [[4]])
    
    def test_rdivide_reverse_operator(self):
        """Test rdivide using reverse / operator"""
        I = Interval(np.array([[2]]), np.array([[4]]))
        result = 8 / I  # This should call __rtruediv__
        
        assert np.allclose(result.inf, [[2]])
        assert np.allclose(result.sup, [[4]])
    
    def test_rdivide_size_mismatch_error(self):
        """Test rdivide with size mismatch should raise error"""
        I = Interval(np.array([[2]]), np.array([[4]]))
        divisor = np.array([[1, 2], [3, 4]])  # Wrong size
        
        with pytest.raises(CORAerror):
            I.rdivide(divisor) 