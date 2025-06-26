"""
test_isequal - unit test function for fullspace isequal

Tests the equality comparison functionality for fullspace objects.

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
from cora_python.contSet.capsule.capsule import Capsule


class TestFullspaceIsequal:
    """Test class for fullspace isequal method"""

    def test_isequal_with_itself(self):
        """Test equality with itself"""
        n = 2
        fs = Fullspace(n)
        
        # Compare with itself
        assert fs.isequal(fs)

    def test_isequal_with_same_dimension(self):
        """Test equality with another fullspace of same dimension"""
        n = 2
        fs1 = Fullspace(n)
        fs2 = Fullspace(n)
        
        # Different objects, same dimension
        assert fs1.isequal(fs2)

    def test_isequal_with_zonotope(self):
        """Test inequality with zonotope"""
        n = 2
        fs = Fullspace(n)
        Z = Zonotope(np.zeros((n, 1)), np.eye(n))
        
        # Should not be equal
        assert not fs.isequal(Z)

    def test_isequal_with_unbounded_interval(self):
        """Test equality with unbounded interval"""
        n = 2
        fs = Fullspace(n)
        I = Interval(-np.inf * np.ones(n), np.inf * np.ones(n))
        
        # Unbounded interval represents fullspace
        assert fs.isequal(I)

    def test_isequal_with_bounded_interval(self):
        """Test inequality with bounded interval"""
        n = 2
        fs = Fullspace(n)
        I = Interval(np.array([1, 2]), np.array([3, 4]))
        
        # Bounded interval is not fullspace
        assert not fs.isequal(I)

    def test_isequal_with_capsule(self):
        """Test inequality with capsule"""
        n = 2
        fs = Fullspace(n)
        C = Capsule(np.array([1, 1]), np.array([-1, 1]), 0.5)
        
        # Should not be equal
        assert not fs.isequal(C)

    def test_isequal_with_numeric_vector(self):
        """Test inequality with numeric vector"""
        n = 2
        fs = Fullspace(n)
        p = np.array([2, 1])
        
        # Point is not fullspace
        assert not fs.isequal(p)

    def test_isequal_different_dimensions(self):
        """Test inequality with different dimensions"""
        fs1 = Fullspace(2)
        fs2 = Fullspace(3)
        
        # Different dimensions
        assert not fs1.isequal(fs2)

    def test_isequal_zero_dimension(self):
        """Test equality for zero-dimensional fullspace"""
        fs1 = Fullspace(0)
        fs2 = Fullspace(0)
        
        # Both zero-dimensional
        assert fs1.isequal(fs2)

    def test_isequal_one_dimension(self):
        """Test equality for one-dimensional fullspace"""
        fs1 = Fullspace(1)
        fs2 = Fullspace(1)
        I_unbounded = Interval(np.array([-np.inf]), np.array([np.inf]))
        
        # 1D fullspace equals 1D unbounded interval
        assert fs1.isequal(fs2)
        assert fs1.isequal(I_unbounded)

    def test_isequal_with_partially_bounded_interval(self):
        """Test inequality with partially bounded interval"""
        n = 2
        fs = Fullspace(n)
        I = Interval(np.array([-np.inf, 2]), np.array([9, np.inf]))
        
        # Partially bounded interval is not fullspace
        assert not fs.isequal(I) 