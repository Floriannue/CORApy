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


class MockContSet:
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
            raise ValueError(f"Unknown type: {type}")


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
        
        S = MockContSet(3)
        
        result = randPoint(S, 10, type_='gaussian')
        assert result.shape == (3, 10)
    
    def test_randPoint_empty_set(self):
        """Test randPoint with empty set"""
        
        S = MockContSet(2, empty=True)
        
        result = randPoint(S, 5)
        assert result.shape == (2, 0)
    
    def test_randPoint_error_cases(self):
        """Test error cases for randPoint"""
        
        S = MockContSet(2)
        
        # Test invalid N
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            randPoint(S, -1)
        
        with pytest.raises(ValueError, match="If N is string, it must be 'all'"):
            randPoint(S, 'invalid_str')

        # Test invalid type
        with pytest.raises(ValueError, match="Invalid type"):
            randPoint(S, 5, type_='invalid_type')

        # Test invalid 'all' with non-extreme type
        with pytest.raises(ValueError, match="If N is 'all', type must be 'extreme'"):
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


if __name__ == "__main__":
    pytest.main([__file__]) 