"""
test_plus - unit test function for fullspace plus

Tests the Minkowski sum functionality for fullspace objects.

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       05-April-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace.fullspace import Fullspace
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.emptySet.emptySet import EmptySet


class TestFullspacePlus:
    """Test class for fullspace plus method"""

    def test_plus_with_vector(self):
        """Test plus operation with vector"""
        n = 2
        fs = Fullspace(n)
        
        # Init vector
        p = np.array([2, 1])
        fs_ = fs.plus(p)
        
        # Fullspace + vector = fullspace
        assert fs_.isequal(fs)

    def test_plus_with_empty_vector(self):
        """Test plus operation with empty vector"""
        n = 2
        fs = Fullspace(n)
        
        # Empty vector
        p = np.empty((n, 0))
        fs_ = fs.plus(p)
        
        # Should result in empty set
        assert fs_.isequal(EmptySet(n))

    def test_plus_with_zonotope(self):
        """Test plus operation with zonotope"""
        n = 2
        fs = Fullspace(n)
        
        # Init zonotope
        Z = Zonotope(np.zeros((n, 1)), np.eye(n))
        fs_ = fs.plus(Z)
        
        # Fullspace + zonotope = fullspace
        assert fs_.isequal(fs)

    def test_plus_with_interval(self):
        """Test plus operation with interval"""
        n = 2
        fs = Fullspace(n)
        
        # Init interval
        I = Interval(np.array([-2, 1]), np.array([np.inf, 3]))
        fs_ = fs.plus(I)
        
        # Fullspace + interval = fullspace
        assert fs_.isequal(fs)

    def test_plus_with_empty_set(self):
        """Test plus operation with empty set"""
        n = 2
        fs = Fullspace(n)
        
        # Init emptySet
        O = EmptySet(n)
        fs_ = fs.plus(O)
        
        # Fullspace + emptySet = emptySet
        assert fs_.isequal(O)

    def test_plus_with_bounded_interval(self):
        """Test plus operation with bounded interval"""
        n = 2
        fs = Fullspace(n)
        
        # Bounded interval
        I = Interval(np.array([1, 2]), np.array([3, 4]))
        fs_ = fs.plus(I)
        
        # Fullspace + bounded interval = fullspace
        assert fs_.isequal(fs)

    def test_plus_with_unbounded_interval(self):
        """Test plus operation with unbounded interval"""
        n = 2
        fs = Fullspace(n)
        
        # Unbounded interval
        I = Interval(-np.inf * np.ones(n), np.inf * np.ones(n))
        fs_ = fs.plus(I)
        
        # Fullspace + unbounded interval = fullspace
        assert fs_.isequal(fs)

    def test_plus_with_fullspace(self):
        """Test plus operation with another fullspace"""
        n = 2
        fs1 = Fullspace(n)
        fs2 = Fullspace(n)
        
        fs_ = fs1.plus(fs2)
        
        # Fullspace + fullspace = fullspace
        assert fs_.isequal(fs1)

    def test_plus_dimension_mismatch(self):
        """Test plus operation with dimension mismatch"""
        fs = Fullspace(2)
        p_wrong_dim = np.array([1, 2, 3])  # 3D vector
        
        # Should raise error for dimension mismatch
        with pytest.raises((ValueError, Exception)):
            fs.plus(p_wrong_dim)

    def test_plus_zero_vector(self):
        """Test plus operation with zero vector"""
        n = 3
        fs = Fullspace(n)
        
        # Zero vector
        zero_vec = np.zeros(n)
        fs_ = fs.plus(zero_vec)
        
        # Fullspace + zero = fullspace
        assert fs_.isequal(fs)

    def test_plus_multiple_vectors(self):
        """Test plus operation with multiple vectors (matrix)"""
        n = 2
        fs = Fullspace(n)
        
        # Multiple vectors as matrix
        points = np.array([[1, 2], [3, 4]])  # 2x2 matrix
        fs_ = fs.plus(points)
        
        # Should still result in fullspace
        assert fs_.isequal(fs)

    def test_plus_large_vector(self):
        """Test plus operation with large vector"""
        n = 2
        fs = Fullspace(n)
        
        # Very large vector
        large_vec = np.array([1e10, -1e10])
        fs_ = fs.plus(large_vec)
        
        # Fullspace + large vector = fullspace
        assert fs_.isequal(fs) 