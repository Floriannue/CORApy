"""
test_zonotope_plus - unit test function of plus method

This module tests the plus method (Minkowski addition) for zonotope objects.

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       26-July-2016 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestZonotopePlus:
    """Test class for zonotope plus method"""
    
    def test_zonotope_plus_zonotope(self):
        """Test Minkowski addition of two zonotopes"""
        # Test case from MATLAB unit test
        Z1 = Zonotope(np.array([[-4], [1]]), np.array([[-3, -2, -1], [2, 3, 4]]))
        Z2 = Zonotope(np.array([[1], [-1]]), np.array([[10], [-10]]))
        
        Z_plus = Z1.plus(Z2)
        
        # Expected result
        c_plus = np.array([[-3], [0]])
        G_plus = np.array([[-3, -2, -1, 10], [2, 3, 4, -10]])
        
        assert np.allclose(Z_plus.c, c_plus)
        assert np.allclose(Z_plus.G, G_plus)
    
    def test_zonotope_plus_zonotope_operator(self):
        """Test Minkowski addition using + operator"""
        Z1 = Zonotope(np.array([[-4], [1]]), np.array([[-3, -2, -1], [2, 3, 4]]))
        Z2 = Zonotope(np.array([[1], [-1]]), np.array([[10], [-10]]))
        
        Z_plus = Z1 + Z2
        
        # Expected result
        c_plus = np.array([[-3], [0]])
        G_plus = np.array([[-3, -2, -1, 10], [2, 3, 4, -10]])
        
        assert np.allclose(Z_plus.c, c_plus)
        assert np.allclose(Z_plus.G, G_plus)
    
    def test_zonotope_plus_vector(self):
        """Test addition of zonotope with vector"""
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        v = np.array([[3], [-1]])
        
        Z_plus = Z.plus(v)
        
        # Expected: center shifts by vector, generators unchanged
        expected_c = np.array([[4], [1]])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_zonotope_plus_vector_operator(self):
        """Test addition of zonotope with vector using + operator"""
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        v = np.array([[3], [-1]])
        
        Z_plus = Z + v
        
        # Expected: center shifts by vector, generators unchanged
        expected_c = np.array([[4], [1]])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_zonotope_plus_scalar(self):
        """Test addition of zonotope with scalar"""
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        s = 5
        
        Z_plus = Z.plus(s)
        
        # Expected: center shifts by scalar, generators unchanged
        expected_c = np.array([[6], [7]])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_zonotope_plus_empty(self):
        """Test Minkowski addition with empty set"""
        Z1 = Zonotope(np.array([[-4], [1]]), np.array([[-3, -2, -1], [2, 3, 4]]))
        Z_empty = Zonotope.empty(2)
        
        Z_plus = Z1.plus(Z_empty)
        
        # Result should be empty
        assert Z_plus.isemptyobject()
    
    def test_empty_plus_zonotope(self):
        """Test Minkowski addition of empty set with zonotope"""
        Z1 = Zonotope(np.array([[-4], [1]]), np.array([[-3, -2, -1], [2, 3, 4]]))
        Z_empty = Zonotope.empty(2)
        
        Z_plus = Z_empty.plus(Z1)
        
        # Result should be empty
        assert Z_plus.isemptyobject()
    
    def test_zonotope_plus_no_generators(self):
        """Test addition of zonotopes with no generators"""
        Z1 = Zonotope(np.array([[1], [2]]))
        Z2 = Zonotope(np.array([[3], [-1]]))
        
        Z_plus = Z1.plus(Z2)
        
        # Expected: centers add, no generators
        expected_c = np.array([[4], [1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert Z_plus.G.shape[1] == 0  # No generators
    
    def test_zonotope_plus_mixed_generators(self):
        """Test addition where one zonotope has no generators"""
        Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([[3], [-1]]))
        
        Z_plus = Z1.plus(Z2)
        
        # Expected: centers add, generators from Z1 only
        expected_c = np.array([[4], [1]])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_1d_zonotope_plus(self):
        """Test addition of 1D zonotopes"""
        Z1 = Zonotope(np.array([[2]]), np.array([[1, -1]]))
        Z2 = Zonotope(np.array([[3]]), np.array([[0.5]]))
        
        Z_plus = Z1.plus(Z2)
        
        # Expected result
        expected_c = np.array([[5]])
        expected_G = np.array([[1, -1, 0.5]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_high_dimensional_plus(self):
        """Test addition of high-dimensional zonotopes"""
        n = 10
        c1 = np.random.randn(n, 1)
        G1 = np.random.randn(n, 5)
        c2 = np.random.randn(n, 1)
        G2 = np.random.randn(n, 3)
        
        Z1 = Zonotope(c1, G1)
        Z2 = Zonotope(c2, G2)
        
        Z_plus = Z1.plus(Z2)
        
        # Expected result
        expected_c = c1 + c2
        expected_G = np.hstack([G1, G2])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_dimension_mismatch(self):
        """Test addition with dimension mismatch raises error"""
        Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))  # 2D
        Z2 = Zonotope(np.array([[1], [2], [3]]), np.array([[1, 0], [0, 1], [1, 0]]))  # 3D
        
        with pytest.raises(CORAerror):
            Z1.plus(Z2)
    
    def test_vector_dimension_mismatch(self):
        """Test addition with vector of wrong dimension raises error"""
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))  # 2D
        v = np.array([[1], [2], [3]])  # 3D vector
        
        with pytest.raises(CORAerror):
            Z.plus(v)

    def test_zonotope_plus_interval(self):
        """Test plus operation between zonotope and interval"""
        from cora_python.contSet import Interval
        
        # Create zonotope
        Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0], [0, 1]]))
        
        # Create interval
        I = Interval([-1, -1], [1, 1])
        
        # Test Z + I (should delegate to I + Z due to higher precedence of Interval)
        result = Z + I
        assert isinstance(result, Interval)
        
        # Expected: same as I + Z due to precedence delegation
        expected = Interval([-1, -2], [3, 2])
        assert result == expected
    
    def test_interval_plus_zonotope(self):
        """Test plus operation between interval and zonotope (reverse order)"""
        from cora_python.contSet import Interval
        
        # Create zonotope
        Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0], [0, 1]]))
        
        # Create interval
        I = Interval([-1, -1], [1, 1])
        
        # Test I + Z (should use Interval's plus method due to higher precedence)
        result = I + Z
        assert isinstance(result, Interval)
        
        # Expected: zonotope Z as interval is [[0,-1], [2,1]]
        # I + Z = [[-1,-1], [1,1]] + [[0,-1], [2,1]] = [[-1,-2], [3,2]]
        expected = Interval([-1, -2], [3, 2])
        assert result == expected
    
    def test_zonotope_plus_interval_different_shapes(self):
        """Test plus operation with different shaped zonotopes and intervals"""
        from cora_python.contSet import Interval
        
        # Create zonotope with specific center and generators
        Z = Zonotope(np.array([[2], [-1]]), np.array([[0.5, 1], [1, 0.5]]))
        
        # Create interval
        I = Interval([0, -2], [1, 0])
        
        # Test Z + I (should delegate to I + Z due to precedence)
        result = Z + I
        assert isinstance(result, Interval)
        
        # Expected: same calculation as I + Z
        expected = Interval([0.5, -4.5], [4.5, 0.5])
        assert result == expected
    
    def test_zonotope_plus_interval_1d(self):
        """Test plus operation with 1D zonotope and interval"""
        from cora_python.contSet import Interval
        
        # Create 1D zonotope
        Z = Zonotope(np.array([[3]]), np.array([[1, 2]]))
        
        # Create 1D interval
        I = Interval([-1], [2])
        
        # Test Z + I (should delegate to I + Z due to precedence)
        result = Z + I
        assert isinstance(result, Interval)
        
        # Expected: same calculation as I + Z
        expected = Interval([-1], [8])
        assert result == expected
    
    def test_zonotope_plus_interval_no_generators(self):
        """Test plus operation with zonotope with no generators and interval"""
        from cora_python.contSet import Interval
        
        # Create zonotope with no generators (point)
        Z = Zonotope(np.array([[1], [2]]))
        
        # Create interval
        I = Interval([-1, 0], [1, 1])
        
        # Test Z + I (should delegate to I + Z due to precedence)
        result = Z + I
        assert isinstance(result, Interval)
        
        # Expected: same calculation as I + Z
        expected = Interval([0, 2], [2, 3])
        assert result == expected
    
    def test_zonotope_plus_interval_zero_radius(self):
        """Test plus operation with interval that has zero radius in some dimensions"""
        from cora_python.contSet import Interval
        
        # Create zonotope
        Z = Zonotope(np.array([[1], [1]]), np.array([[1, 0], [0, 1]]))
        
        # Create interval with zero radius in first dimension
        I = Interval([2, -1], [2, 1])
        
        # Test Z + I (should delegate to I + Z due to precedence)
        result = Z + I
        assert isinstance(result, Interval)
        
        # Expected: same calculation as I + Z
        expected = Interval([2, -1], [4, 3])
        assert result == expected
    
    def test_zonotope_plus_interval_empty(self):
        """Test plus operation with empty interval and zonotope"""
        from cora_python.contSet import Interval
        
        # Create zonotope
        Z = Zonotope(np.array([[1], [0]]), np.array([[1, 0], [0, 1]]))
        
        # Create empty interval
        I = Interval.empty(2)
        
        # Test Z + I (should delegate to I + Z and result in empty interval)
        result = Z + I
        assert result.representsa_('emptySet')
        
        # Test I + Z (should also result in empty)
        result = I + Z
        assert result.representsa_('emptySet')
    
    def test_zonotope_interval_precedence(self):
        """Test that precedence system works correctly for mixed operations"""
        from cora_python.contSet import Interval
        
        # Create zonotope and interval
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        I = Interval([-1, -1], [1, 1])
        
        # Both operations should delegate to interval's plus method due to higher precedence
        result1 = Z + I
        result2 = I + Z
        
        # Both should return intervals and be equal
        assert isinstance(result1, Interval)
        assert isinstance(result2, Interval)
        assert result1 == result2


if __name__ == "__main__":
    test = TestZonotopePlus()
    test.test_zonotope_plus_zonotope()
    test.test_zonotope_plus_zonotope_operator()
    test.test_zonotope_plus_vector()
    test.test_zonotope_plus_vector_operator()
    test.test_zonotope_plus_scalar()
    test.test_zonotope_plus_empty()
    test.test_empty_plus_zonotope()
    test.test_zonotope_plus_no_generators()
    test.test_zonotope_plus_mixed_generators()
    test.test_1d_zonotope_plus()
    test.test_high_dimensional_plus()
    test.test_dimension_mismatch()
    test.test_vector_dimension_mismatch()
    test.test_zonotope_plus_interval()
    test.test_interval_plus_zonotope()
    test.test_zonotope_plus_interval_different_shapes()
    test.test_zonotope_plus_interval_1d()
    test.test_zonotope_plus_interval_no_generators()
    test.test_zonotope_plus_interval_zero_radius()
    test.test_zonotope_plus_interval_empty()
    test.test_zonotope_interval_precedence()
    print("All tests passed!") 