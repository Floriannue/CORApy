"""
test_interval_plus - unit test function for interval plus operation

This module tests the plus operation (Minkowski sum) for intervals,
including addition with other intervals and numeric values.

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


class TestIntervalPlus:
    """Test class for interval plus operation"""
    
    def test_plus_empty_Interval(self):
        """Test plus operation with empty intervals"""
        I = Interval.empty(1)
        v = 1
        I_plus = I + v
        assert I_plus.representsa_('emptySet')
        
        # Reverse operation
        I_plus = v + I
        assert I_plus.representsa_('emptySet')
    
    def test_plus_bounded_interval_numeric(self):
        """Test plus operation between bounded interval and numeric"""
        tol = 1e-9
        
        v = np.array([2, 1])
        I = Interval([-2, -1], [3, 4])
        I_plus = v + I
        I_true = Interval([0, 0], [5, 5])
        
        assert I_plus == I_true
        
        # Reverse operation
        I_plus = I + v
        assert I_plus == I_true
    
    def test_plus_unbounded_interval_numeric(self):
        """Test plus operation with unbounded intervals"""
        tol = 1e-9
        
        I = Interval([-np.inf], [2])
        v = 1
        I_plus = I + v
        I_true = Interval([-np.inf], [3])
        
        assert np.isinf(I_plus.inf[0]) and I_plus.inf[0] < 0
        assert np.isclose(I_plus.sup[0], 3, atol=tol)
        
        # Reverse operation
        I_plus = v + I
        assert np.isinf(I_plus.inf[0]) and I_plus.inf[0] < 0
        assert np.isclose(I_plus.sup[0], 3, atol=tol)
    
    def test_plus_bounded_intervals(self):
        """Test plus operation between two bounded intervals"""
        tol = 1e-9
        
        I1 = Interval([-2, -1], [3, 4])
        I2 = Interval([-1, -3], [1, -1])
        I_plus = I1 + I2
        I_true = Interval([-3, -4], [4, 3])
        
        assert I_plus == I_true
        
        # Reverse operation (should be commutative)
        I_plus = I2 + I1
        assert I_plus == I_true
    
    def test_plus_unbounded_intervals(self):
        """Test plus operation between unbounded intervals"""
        tol = 1e-9
        
        I1 = Interval([-np.inf, -2], [2, 4])
        I2 = Interval([-1, 0], [1, np.inf])
        I_plus = I1 + I2
        I_true = Interval([-np.inf, -2], [3, np.inf])
        
        # Check bounds
        assert np.isinf(I_plus.inf[0]) and I_plus.inf[0] < 0
        assert np.isclose(I_plus.inf[1], -2, atol=tol)
        assert np.isclose(I_plus.sup[0], 3, atol=tol)
        assert np.isinf(I_plus.sup[1]) and I_plus.sup[1] > 0
        
        # Reverse operation
        I_plus = I2 + I1
        assert np.isinf(I_plus.inf[0]) and I_plus.inf[0] < 0
        assert np.isclose(I_plus.inf[1], -2, atol=tol)
        assert np.isclose(I_plus.sup[0], 3, atol=tol)
        assert np.isinf(I_plus.sup[1]) and I_plus.sup[1] > 0
    
    def test_plus_interval_matrix_numeric(self):
        """Test plus operation between interval matrix and numeric"""
        tol = 1e-9
        
        I = Interval([[-2, -1], [0, 2]], [[3, 5], [2, 3]])
        v = 2
        I_plus = I + v
        I_true = Interval([[0, 1], [2, 4]], [[5, 7], [4, 5]])
        
        assert I_plus == I_true
        
        # Reverse operation
        I_plus = v + I
        assert I_plus == I_true
    
    def test_plus_scalar_operations(self):
        """Test plus operation with scalar values"""
        I = Interval([-1, 0], [2, 3])
        
        # Add scalar
        I_plus = I + 5
        expected = Interval([4, 5], [7, 8])
        assert I_plus == expected
        
        # Reverse add scalar
        I_plus = 5 + I
        assert I_plus == expected
    
    def test_plus_vector_operations(self):
        """Test plus operation with vector values"""
        I = Interval([-1, 0], [2, 3])
        v = np.array([1, -1])
        
        I_plus = I + v
        expected = Interval([0, -1], [3, 2])
        assert I_plus == expected
        
        # Reverse operation
        I_plus = v + I
        assert I_plus == expected
    
    def test_plus_matrix_operations(self):
        """Test plus operation with matrix intervals"""
        I1 = Interval([[-1, 0], [1, -1]], [[2, 1], [3, 2]])
        I2 = Interval([[0, -1], [-1, 0]], [[1, 0], [2, 1]])
        
        I_plus = I1 + I2
        expected = Interval([[-1, -1], [0, -1]], [[3, 1], [5, 3]])
        assert I_plus == expected
    
    def test_plus_zero_operations(self):
        """Test plus operation with zero"""
        I = Interval([-1, 0], [2, 3])
        
        # Add zero
        I_plus = I + 0
        assert I_plus == I
        
        # Add zero vector
        zero_vec = np.zeros(2)
        I_plus = I + zero_vec
        assert I_plus == I
    
    def test_plus_commutativity(self):
        """Test that plus operation is commutative"""
        I1 = Interval([-2, -1], [3, 4])
        I2 = Interval([1, 0], [2, 1])
        
        result1 = I1 + I2
        result2 = I2 + I1
        assert result1 == result2
        
        # With numeric
        v = np.array([1, -1])
        result1 = I1 + v
        result2 = v + I1
        assert result1 == result2
    
    def test_plus_associativity(self):
        """Test associativity of plus operation"""
        I1 = Interval([-1, 0], [1, 2])
        I2 = Interval([0, -1], [2, 1])
        I3 = Interval([-1, -1], [1, 1])
        
        # (I1 + I2) + I3
        result1 = (I1 + I2) + I3
        
        # I1 + (I2 + I3)
        result2 = I1 + (I2 + I3)
        
        assert result1 == result2

    def test_plus_interval_zonotope(self):
        """Test plus operation between interval and zonotope"""
        from cora_python.contSet import Zonotope
        
        # Create interval
        I = Interval([-1, -1], [1, 1])
        
        # Create zonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Test I + Z (should use Interval's plus method due to higher precedence)
        result = I + Z
        assert isinstance(result, Interval)
        
        # Expected: interval bounds should be [-2, -2] to [2, 2]
        # Since zonotope Z has center [0,0] and generators [[1,0],[0,1]]
        # its interval representation is [[-1,-1], [1,1]]
        # So I + Z = [[-1,-1], [1,1]] + [[-1,-1], [1,1]] = [[-2,-2], [2,2]]
        expected = Interval([-2, -2], [2, 2])
        assert result == expected
    
    def test_plus_zonotope_interval(self):
        """Test plus operation between zonotope and interval (reverse order)"""
        from cora_python.contSet import Zonotope
        
        # Create interval
        I = Interval([-1, -1], [1, 1])
        
        # Create zonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Test Z + I (should delegate to I + Z due to precedence)
        result = Z + I
        assert isinstance(result, Interval)
        
        # Should give same result as I + Z
        expected = Interval([-2, -2], [2, 2])
        assert result == expected
    
    def test_plus_interval_zonotope_different_shapes(self):
        """Test plus operation with different shaped intervals and zonotopes"""
        from cora_python.contSet import Zonotope
        
        # Create interval with different bounds
        I = Interval([0, -2], [2, 1])
        
        # Create zonotope with different center and generators
        c = np.array([[1], [-1]])
        G = np.array([[0.5, 0], [0, 0.5]])
        Z = Zonotope(c, G)
        
        # Test I + Z
        result = I + Z
        assert isinstance(result, Interval)
        
        # Zonotope Z has interval representation: center ± sum(abs(generators))
        # center = [1, -1], generators = [[0.5, 0], [0, 0.5]]
        # sum(abs(generators)) = [0.5, 0.5]
        # So Z as interval = [[0.5, -1.5], [1.5, -0.5]]
        # I + Z = [[0, -2], [2, 1]] + [[0.5, -1.5], [1.5, -0.5]] = [[0.5, -3.5], [3.5, 0.5]]
        expected = Interval([0.5, -3.5], [3.5, 0.5])
        assert result == expected
    
    def test_plus_interval_zonotope_1d(self):
        """Test plus operation with 1D interval and zonotope"""
        from cora_python.contSet import Zonotope
        
        # Create 1D interval
        I = Interval([-2], [3])
        
        # Create 1D zonotope
        c = np.array([[1]])
        G = np.array([[2]])
        Z = Zonotope(c, G)
        
        # Test I + Z
        result = I + Z
        assert isinstance(result, Interval)
        
        # Zonotope Z as interval: [1-2, 1+2] = [-1, 3]
        # I + Z = [-2, 3] + [-1, 3] = [-3, 6]
        expected = Interval([-3], [6])
        assert result == expected
    
    def test_plus_interval_zonotope_no_generators(self):
        """Test plus operation with interval and zonotope with no generators"""
        from cora_python.contSet import Zonotope
        
        # Create interval
        I = Interval([-1, 0], [1, 2])
        
        # Create zonotope with no generators (point)
        c = np.array([[2], [1]])
        Z = Zonotope(c)
        
        # Test I + Z
        result = I + Z
        assert isinstance(result, Interval)
        
        # Zonotope Z as interval: [2, 1] (point)
        # I + Z = [[-1, 0], [1, 2]] + [[2], [1]] = [[1, 1], [3, 3]]
        expected = Interval([1, 1], [3, 3])
        assert result == expected
    
    def test_plus_interval_zonotope_empty(self):
        """Test plus operation with empty interval and zonotope"""
        from cora_python.contSet import Zonotope
        
        # Create empty interval
        I = Interval.empty(2)
        
        # Create zonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Test I + Z (should result in empty)
        result = I + Z
        assert result.representsa_('emptySet')
        
        # Test Z + I (should also result in empty)
        result = Z + I
        assert result.representsa_('emptySet')


if __name__ == '__main__':
    pytest.main([__file__]) 
