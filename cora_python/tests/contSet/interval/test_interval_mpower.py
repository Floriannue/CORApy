import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalMpower:
    """Test class for interval mpower method"""
    
    def test_mpower_scalar_positive_exponent(self):
        """Test mpower with scalar interval and positive exponent"""
        # Create a scalar interval
        I = Interval(np.array([[2]]), np.array([[3]]))
        res = I.mpower(2)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # For positive interval [2, 3]^2 = [4, 9]
        expected_inf = np.array([[4]])
        expected_sup = np.array([[9]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_mpower_scalar_negative_base(self):
        """Test mpower with negative scalar interval"""
        # Create a scalar interval with negative values
        I = Interval(np.array([[-3]]), np.array([[-2]]))
        res = I.mpower(2)
        
        # For negative interval [-3, -2]^2 = [4, 9]
        expected_inf = np.array([[4]])
        expected_sup = np.array([[9]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_mpower_scalar_mixed_signs(self):
        """Test mpower with scalar interval crossing zero"""
        # Create a scalar interval crossing zero
        I = Interval(np.array([[-2]]), np.array([[3]]))
        res = I.mpower(2)
        
        # For interval [-2, 3]^2 = [0, 9]
        expected_inf = np.array([[0]])
        expected_sup = np.array([[9]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_mpower_scalar_exponent_zero(self):
        """Test mpower with exponent zero"""
        # Create a scalar interval
        I = Interval(np.array([[2]]), np.array([[3]]))
        res = I.mpower(0)
        
        # Any number to power 0 is 1
        expected_inf = np.array([[1]])
        expected_sup = np.array([[1]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_mpower_scalar_exponent_one(self):
        """Test mpower with exponent one"""
        # Create a scalar interval
        I = Interval(np.array([[2]]), np.array([[3]]))
        res = I.mpower(1)
        
        # Any number to power 1 is itself
        assert res.isequal(I)
    
    def test_mpower_matrix_square(self):
        """Test mpower with square matrix interval"""
        # Create a 2x2 interval matrix
        I = Interval(np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 2]]))
        res = I.mpower(2)
        
        # Verify result is an interval
        assert isinstance(res, Interval)
        
        # Result should be a 2x2 interval
        assert res.inf.shape == (2, 2)
        assert res.sup.shape == (2, 2)
    
    def test_mpower_matrix_exponent_zero(self):
        """Test mpower with matrix and exponent zero"""
        # Create a 2x2 interval matrix
        I = Interval(np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 2]]))
        res = I.mpower(0)
        
        # Should return identity matrix
        expected_identity = np.eye(2)
        assert np.allclose(res.inf, expected_identity)
        assert np.allclose(res.sup, expected_identity)
    
    def test_mpower_matrix_exponent_one(self):
        """Test mpower with matrix and exponent one"""
        # Create a 2x2 interval matrix
        I = Interval(np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 2]]))
        res = I.mpower(1)
        
        # Should return the same matrix
        assert res.isequal(I)
    
    def test_mpower_invalid_exponent(self):
        """Test mpower with invalid exponent"""
        # Create a scalar interval
        I = Interval(np.array([[2]]), np.array([[3]]))
        
        # Non-scalar exponent should raise error
        with pytest.raises(CORAerror):
            I.mpower(np.array([1, 2]))
    
    def test_mpower_non_square_matrix_zero_exponent(self):
        """Test mpower with non-square matrix and zero exponent"""
        # Create a 2x3 interval matrix
        I = Interval(np.array([[1, 0, 1], [0, 1, 0]]), np.array([[2, 1, 2], [1, 2, 1]]))
        
        # Should raise error for non-square matrix with exponent 0
        with pytest.raises(CORAerror):
            I.mpower(0)
    
    def test_mpower_negative_exponent(self):
        """Test mpower with negative exponent"""
        # Create a scalar interval
        I = Interval(np.array([[2]]), np.array([[3]]))
        
        # For scalar intervals, negative exponent should work (delegates to element-wise power)
        res = I.mpower(-1)
        
        # [2, 3]^(-1) = [1/3, 1/2]
        expected_inf = np.array([[1/3]])
        expected_sup = np.array([[1/2]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_mpower_fractional_exponent(self):
        """Test mpower with fractional exponent"""
        # Create a scalar interval
        I = Interval(np.array([[4]]), np.array([[9]]))
        
        # For scalar intervals, fractional exponent should work (delegates to element-wise power)
        res = I.mpower(0.5)
        
        # [4, 9]^(0.5) = [2, 3]
        expected_inf = np.array([[2]])
        expected_sup = np.array([[3]])
        
        assert np.allclose(res.inf, expected_inf)
        assert np.allclose(res.sup, expected_sup)
    
    def test_mpower_matrix_negative_exponent(self):
        """Test mpower with matrix and negative exponent"""
        # Create a 2x2 interval matrix
        I = Interval(np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 2]]))
        
        # Matrix with negative exponent should raise error
        with pytest.raises(CORAerror):
            I.mpower(-1)
    
    def test_mpower_matrix_fractional_exponent(self):
        """Test mpower with matrix and fractional exponent"""
        # Create a 2x2 interval matrix
        I = Interval(np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 2]]))
        
        # Matrix with fractional exponent should raise error
        with pytest.raises(CORAerror):
            I.mpower(0.5)


def test_interval_mpower():
    """Basic test for interval mpower method"""
    # Test with simple scalar interval
    I = Interval(np.array([[2]]), np.array([[3]]))
    res = I.mpower(2)
    
    # Verify result is an interval
    assert isinstance(res, Interval)
    
    # For [2, 3]^2 = [4, 9]
    expected_inf = np.array([[4]])
    expected_sup = np.array([[9]])
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup) 