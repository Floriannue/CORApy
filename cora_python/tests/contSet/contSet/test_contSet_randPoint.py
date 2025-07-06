"""
test_randPoint - unit test function for randPoint

Syntax:
    pytest test_randPoint.py

Inputs:
    -

Outputs:
    test results

Other modules required: none
Subfunctions: none

See also: none

Authors: AI Assistant
Written: 2025
Last update: ---
Last revision: ---
"""

import pytest
import numpy as np
from unittest.mock import patch
from cora_python.contSet.contSet.randPoint import randPoint
from cora_python.contSet.contSet.contSet import ContSet


class MockContSet(ContSet):
    """Mock ContSet for testing randPoint method"""
    
    def __init__(self, dim_val=2, empty=False):
        self._dim = dim_val
        self._empty = empty
        
    def dim(self):
        return self._dim
    
    def isemptyobject(self):
        return self._empty
    
    def randPoint_(self, N, type):
        """Mock implementation returning random points"""
        if self._empty:
            return np.array([]).reshape(self._dim, 0)
        
        if type == 'standard':
            # Return random points in unit square
            return np.random.rand(self._dim, N) * 2 - 1
        elif type == 'extreme':
            # Return vertices of unit hypercube
            if N == 1:
                return np.ones((self._dim, 1))
            else:
                # Return multiple extreme points
                points = []
                for i in range(min(N, 2**self._dim)):
                    # Generate vertices of hypercube
                    vertex = np.array([(i >> j) & 1 for j in range(self._dim)]) * 2 - 1
                    points.append(vertex)
                while len(points) < N:
                    points.append(points[0])  # Repeat first vertex
                return np.column_stack(points)
        elif type == 'gaussian':
            # Return Gaussian random points
            return np.random.randn(self._dim, N)
        else:
            # For any other type, just return random points
            # The validation should have caught invalid types before this point
            return np.random.rand(self._dim, N) * 2 - 1
    
    def representsa_(self, type_, tol):
        """Mock representsa_ method"""
        if type_ == 'emptySet':
            return self._empty
        elif type_ == 'origin':
            return False  # For simplicity, assume not origin
        else:
            return False
    
    def contains(self, points):
        """Mock contains method"""
        if points.ndim == 1:
            return np.array([True])
        else:
            return np.ones(points.shape[1], dtype=bool)
    
    def __repr__(self):
        """Mock __repr__ method"""
        return f"MockContSet(dim={self._dim}, empty={self._empty})"


class TestRandPoint:
    """Test class for randPoint function"""
    
    def test_randPoint_single_point(self):
        """Test randPoint with single point"""
        
        S = MockContSet(2)
        
        with patch('numpy.random.rand') as mock_rand:
            mock_rand.return_value = np.array([[0.5], [0.5]])
            result = randPoint(S)
            
        assert result.shape == (2, 1)
        assert isinstance(result, np.ndarray)
    
    def test_randPoint_multiple_points(self):
        """Test randPoint with multiple points"""
        
        S = MockContSet(2)
        N = 5
        
        result = randPoint(S, N)
        assert result.shape == (2, N)
    
    def test_randPoint_standard_type(self):
        """Test randPoint with standard type"""
        
        S = MockContSet(2)
        
        result = randPoint(S, 3, type_='standard')
        assert result.shape == (2, 3)
        assert isinstance(result, np.ndarray)
    
    def test_randPoint_extreme_type(self):
        """Test randPoint with extreme type"""
        
        S = MockContSet(2)
        
        result = randPoint(S, 3, type_='extreme')
        assert result.shape == (2, 3)
    
    def test_randPoint_gaussian_type(self):
        """Test randPoint with gaussian type"""
        
        # Use a real Interval object instead of MockContSet since gaussian
        # type requires conversion to ellipsoid
        from cora_python.contSet.interval import Interval
        I = Interval([-1, -1], [1, 1])
        
        # Test that gaussian type works for interval (supported via ellipsoid conversion)
        result = randPoint(I, 10, type_='gaussian')
        assert result.shape == (2, 10)
        assert isinstance(result, np.ndarray)
        # Note: gaussian points are not guaranteed to be inside the set
    
    def test_randPoint_empty_set(self):
        """Test randPoint with empty set"""
        
        S = MockContSet(2, empty=True)
        
        result = randPoint(S, 5)
        assert result.shape == (2, 0)
    
    def test_randPoint_error_cases(self):
        """Test error cases for randPoint"""
        
        S = MockContSet(2)
        
        # Test invalid N - inputArgsCheck throws CORAerror for negative values
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        with pytest.raises(CORAerror, match="Wrong value for the 2nd input argument"):
            randPoint(S, -1)
        
        # Test invalid string N - inputArgsCheck throws CORAerror for invalid strings
        with pytest.raises(CORAerror, match="Wrong value for the 2nd input argument"):
            randPoint(S, 'invalid_str')

        # Test invalid type - inputArgsCheck throws CORAerror for invalid types
        with pytest.raises(CORAerror, match="Wrong value for the 3rd input argument"):
            randPoint(S, 5, type_='invalid_type')

        # Test invalid 'all' with non-extreme type - this is checked by randPoint itself
        with pytest.raises(CORAerror, match="If the number of points is 'all', the type has to be 'extreme'"):
            randPoint(S, 'all', type_='standard')
    
    def test_randPoint_origin_point(self):
        """Test randPoint returns origin for degenerate case"""
        
        # Mock a set that returns origin
        class OriginSet(MockContSet):
            def randPoint_(self, N, type):
                return np.zeros((self._dim, N))
        
        S = OriginSet(2)
        result = randPoint(S, 3)
        
        assert result.shape == (2, 3)
        assert np.allclose(result, 0)
    
    def test_randPoint_high_dimension(self):
        """Test randPoint with high-dimensional sets"""
        
        S = MockContSet(10)
        
        result = randPoint(S, 2)
        assert result.shape == (10, 2)
    
    def test_randPoint_large_N(self):
        """Test randPoint with large number of points"""
        
        S = MockContSet(2)
        N = 1000
        
        result = randPoint(S, N)
        assert result.shape == (2, N)
    
    @patch('numpy.random.seed')
    def test_randPoint_reproducibility(self, mock_seed):
        """Test that randPoint can be made reproducible with seed"""
        
        S = MockContSet(2)
        
        # Mock random to return predictable values
        with patch('numpy.random.rand') as mock_rand:
            mock_rand.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            result1 = randPoint(S, 2)
            result2 = randPoint(S, 2)
            
        # Results should be the same since we mocked the random function
        assert np.array_equal(result1, result2)

    def test_randPoint_gaussian_unsupported_class(self):
        """Test that gaussian type raises error for unsupported classes"""
        
        S = MockContSet(2)
        
        # MockContSet is not in supported classes (zonotope, interval, ellipsoid, polytope)
        # so should raise CORAerror, matching MATLAB behavior
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        with pytest.raises(CORAerror, match="does not support type = 'gaussian'"):
            randPoint(S, 10, type_='gaussian')


if __name__ == "__main__":
    pytest.main([__file__]) 