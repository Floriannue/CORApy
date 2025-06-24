"""
test_contains - unit test function for fullspace contains

Tests the containment check functionality for fullspace objects.

Authors:       Mark Wetzlinger, Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       05-April-2023
Last update:   20-January-2025 (AK, added general containment checks)
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace.fullspace import Fullspace
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.emptySet.emptySet import EmptySet
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope


class TestFullspaceContains:
    """Test class for fullspace contains method"""

    def test_contains_point(self):
        """Test containment of a point"""
        n = 2
        fs = Fullspace(n)
        
        # Init point
        p = np.array([2, 1])
        assert fs.contains(p)

    def test_contains_zonotope(self):
        """Test containment of zonotope"""
        n = 2
        fs = Fullspace(n)
        
        # Init zonotope
        Z = Zonotope(np.array([2, 1]), np.eye(n))
        assert fs.contains(Z)

    def test_contains_ellipsoid(self):
        """Test containment of ellipsoid"""
        n = 2
        fs = Fullspace(n)
        
        # Init ellipsoid
        E = Ellipsoid(np.eye(n), np.ones(n))
        assert fs.contains(E)

    def test_contains_empty_set(self):
        """Test containment of empty set"""
        n = 2
        fs = Fullspace(n)
        
        # Empty set
        O = EmptySet(n)
        assert fs.contains(O)

    def test_contains_interval(self):
        """Test containment of interval"""
        n = 2
        fs = Fullspace(n)
        
        # Bounded interval
        I = Interval(np.array([1, 2]), np.array([3, 4]))
        assert fs.contains(I)
        
        # Unbounded interval
        I_unbounded = Interval(np.array([-np.inf, 2]), np.array([9, np.inf]))
        assert fs.contains(I_unbounded)

    def test_contains_fullspace(self):
        """Test containment of another fullspace"""
        n = 2
        fs1 = Fullspace(n)
        fs2 = Fullspace(n)
        
        assert fs1.contains(fs2)

    def test_contains_polytope(self):
        """Test containment of polytope"""
        n = 2
        fs = Fullspace(n)
        
        # Simple unit box polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        assert fs.contains(P)

    def test_dimension_mismatch(self):
        """Test containment with dimension mismatch"""
        fs = Fullspace(2)
        p_wrong_dim = np.array([1, 2, 3])  # 3D point
        
        # Should raise error or return False for dimension mismatch
        with pytest.raises((ValueError, Exception)):
            fs.contains(p_wrong_dim)

    def test_contains_multiple_points(self):
        """Test containment of multiple points"""
        n = 2
        fs = Fullspace(n)
        
        # Multiple points as matrix
        points = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        result = fs.contains(points)
        
        # All points should be contained
        if isinstance(result, np.ndarray):
            assert np.all(result)
        else:
            assert result

    def test_contains_zero_point(self):
        """Test containment of zero point"""
        n = 3
        fs = Fullspace(n)
        
        zero_point = np.zeros(n)
        assert fs.contains(zero_point)

    def test_contains_large_coordinates(self):
        """Test containment of points with large coordinates"""
        n = 2
        fs = Fullspace(n)
        
        # Very large coordinates
        large_point = np.array([1e10, -1e10])
        assert fs.contains(large_point)
        
        # Infinite coordinates
        inf_point = np.array([np.inf, -np.inf])
        assert fs.contains(inf_point) 