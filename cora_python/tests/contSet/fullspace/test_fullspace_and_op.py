"""
test_and_op - unit test function for fullspace and operation

Tests the intersection functionality for fullspace objects.

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant  
Written:       25-April-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace.fullspace import Fullspace
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.emptySet.emptySet import EmptySet
from cora_python.contSet.interval.interval import Interval


class TestFullspaceAndOp:
    """Test class for fullspace and operation"""

    def test_fullspace_with_itself(self):
        """Test intersection with itself"""
        n = 2
        fs = Fullspace(n)
        
        # Intersection with itself
        result = fs.and_op(fs)
        assert result.isequal(fs)

    def test_fullspace_with_zonotope(self):
        """Test intersection with zonotope"""
        n = 2
        fs = Fullspace(n)
        Z = Zonotope(np.zeros((n, 1)), np.eye(n))
        
        # Intersection with zonotope returns the zonotope
        result = fs.and_op(Z)
        assert result.isequal(Z)

    def test_fullspace_with_emptyset(self):
        """Test intersection with empty set"""
        n = 2
        fs = Fullspace(n)
        O = EmptySet(n)
        
        # Intersection with empty set returns empty set
        result = fs.and_op(O)
        assert result.isequal(O)

    def test_fullspace_with_unbounded_interval(self):
        """Test intersection with unbounded interval"""
        n = 2
        fs = Fullspace(n)
        I = Interval(np.array([-np.inf, 2]), np.array([9, np.inf]))
        
        # Intersection with interval returns the interval
        result = fs.and_op(I)
        assert result.isequal(I)

    def test_fullspace_with_numeric_vector(self):
        """Test intersection with numeric vector"""
        n = 2
        fs = Fullspace(n)
        p = np.array([1, -1])
        
        # Intersection with point returns the point
        result = fs.and_op(p)
        assert np.allclose(result, p)

    def test_dimension_mismatch(self):
        """Test intersection with different dimensions"""
        fs1 = Fullspace(2)
        fs2 = Fullspace(3)
        
        # Should raise error for dimension mismatch
        with pytest.raises((ValueError, Exception)):
            fs1.and_op(fs2)

    def test_fullspace_with_bounded_interval(self):
        """Test intersection with bounded interval"""
        n = 2
        fs = Fullspace(n)
        I = Interval(np.array([1, 2]), np.array([3, 4]))
        
        # Intersection with bounded interval returns the interval
        result = fs.and_op(I)
        assert result.isequal(I)

    def test_empty_fullspace(self):
        """Test intersection with zero-dimensional fullspace"""
        fs0 = Fullspace(0)
        
        # Zero-dimensional case
        result = fs0.and_op(fs0)
        assert result.isequal(fs0) 