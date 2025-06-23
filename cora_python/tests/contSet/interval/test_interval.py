"""
test_interval - unit test function for interval class

This module tests the basic functionality of the interval class including
construction, properties, and basic operations.

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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestInterval:
    """Test class for interval functionality"""
    
    def test_constructor_basic(self):
        """Test basic interval construction"""
        # Point interval
        I = Interval([1, 2])
        assert np.allclose(I.inf, [1, 2])
        assert np.allclose(I.sup, [1, 2])
        
        # Interval with bounds
        I = Interval([-2, -1], [3, 4])
        assert np.allclose(I.inf, [-2, -1])
        assert np.allclose(I.sup, [3, 4])
    
    def test_constructor_copy(self):
        """Test copy constructor"""
        I1 = Interval([-2, -1], [3, 4])
        I2 = Interval(I1)
        
        assert I1 == I2
        assert I1.inf is not I2.inf  # Should be a copy, not reference
        assert I1.sup is not I2.sup
    
    def test_constructor_errors(self):
        """Test constructor error cases"""
        # No arguments
        with pytest.raises(CORAerror):
            Interval()
        
        # Lower bound larger than upper bound
        with pytest.raises(CORAerror):
            Interval([3, 4], [1, 2])
        
        # Different dimensions
        with pytest.raises(CORAerror):
            Interval([1, 2], [3])
    
    def test_empty_interval(self):
        """Test empty interval creation and properties"""
        I_empty = Interval.empty(2)
        assert I_empty.is_empty()
        assert I_empty.dim() == 2  # Empty interval with dimension 2 should return 2
        
        # Empty interval with dimension 0
        I_empty0 = Interval.empty(0)
        assert I_empty0.is_empty()
        assert I_empty0.dim() == 0
    
    def test_fullspace_interval(self):
        """Test fullspace interval creation"""
        I_inf = Interval.Inf(2)
        assert np.all(np.isinf(I_inf.inf))
        assert np.all(np.isinf(I_inf.sup))
        assert np.all(I_inf.inf < 0)
        assert np.all(I_inf.sup > 0)
        assert I_inf.dim() == 2
    
    def test_origin_interval(self):
        """Test origin interval creation"""
        I_origin = Interval.origin(3)
        assert np.allclose(I_origin.inf, [0, 0, 0])
        assert np.allclose(I_origin.sup, [0, 0, 0])
        assert I_origin.dim() == 3
        assert I_origin.representsa_('origin')
    
    def test_properties(self):
        """Test interval properties"""
        # Regular interval
        I = Interval([-2, -1], [3, 4])
        assert I.dim() == 2
        assert not I.is_empty()
        assert I.precedence == 120
        
        # Point interval
        I_point = Interval([1, 2], [1, 2])
        assert I_point.representsa_('point')
        
        # Origin interval
        I_origin = Interval([0, 0], [0, 0])
        assert I_origin.representsa_('origin')
    
    def test_contains(self):
        """Test contains method"""
        I = Interval([-2, -1], [3, 4])
        
        # Point inside
        assert I.contains([0, 0])
        assert I.contains([3, 4])  # Boundary
        assert I.contains([-2, -1])  # Boundary
        
        # Point outside
        assert not I.contains([4, 0])
        assert not I.contains([0, 5])
        assert not I.contains([-3, 0])
        
        # Empty interval contains nothing
        I_empty = Interval.empty(2)
        assert not I_empty.contains([0, 0])
    
    def test_equality(self):
        """Test equality comparison"""
        I1 = Interval([-2, -1], [3, 4])
        I2 = Interval([-2, -1], [3, 4])
        I3 = Interval([-1, -1], [3, 4])
        
        assert I1 == I2
        assert I1 != I3
        
        # Empty intervals
        I_empty1 = Interval.empty(2)
        I_empty2 = Interval.empty(2)
        assert I_empty1 == I_empty2
        
        # Empty vs non-empty
        assert I1 != I_empty1
    
    def test_string_representation(self):
        """Test string representation"""
        I = Interval([-2, -1], [3, 4])
        str_repr = str(I)
        assert "Interval object" in str_repr
        assert "[-2, 3]" in str_repr
        assert "[-1, 4]" in str_repr
        
        # Empty interval
        I_empty = Interval.empty(2)
        str_empty = str(I_empty)
        assert "empty set" in str_empty
    
    def test_matrix_intervals(self):
        """Test matrix interval operations"""
        # 2x2 matrix interval
        inf_mat = np.array([[-2, -1], [0, 2]])
        sup_mat = np.array([[3, 5], [2, 3]])
        I = Interval(inf_mat, sup_mat)
        
        assert I.inf.shape == (2, 2)
        assert I.sup.shape == (2, 2)
        assert np.allclose(I.inf, inf_mat)
        assert np.allclose(I.sup, sup_mat)
    
    def test_tolerance_handling(self):
        """Test tolerance in bound checking"""
        # Bounds that are equal within tolerance
        tol = 1e-6
        I = Interval([1.0], [1.0 + tol/2])  # Should be valid
        assert np.allclose(I.inf, [1.0])
        assert np.allclose(I.sup, [1.0 + tol/2])
    
    def test_infinite_bounds(self):
        """Test handling of infinite bounds"""
        # Unbounded interval
        I = Interval([-np.inf, -2], [2, np.inf])
        assert I.dim() == 2
        assert not I.is_empty()
        
        # Contains test with infinite bounds
        assert I.contains([0, 0])
        assert I.contains([-1000, 1000])  # Large values should be contained


if __name__ == '__main__':
    pytest.main([__file__]) 