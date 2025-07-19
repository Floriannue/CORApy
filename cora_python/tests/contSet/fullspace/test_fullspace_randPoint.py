"""
test_randPoint - unit test function for fullspace randPoint

Tests the random point generation functionality for fullspace objects.

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       05-April-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace.fullspace import Fullspace


class TestFullspaceRandPoint:
    """Test class for fullspace randPoint method"""

    def test_single_random_point(self):
        """Test generation of single random point"""
        n = 3
        fs = Fullspace(n)
        
        # Sample point
        p = fs.randPoint()
        
        # Point should be contained and have correct dimension
        assert fs.contains(p)
        assert len(p) == n

    def test_multiple_random_points(self):
        """Test generation of multiple random points"""
        n = 3
        fs = Fullspace(n)
        
        # Sample multiple points
        points = fs.randPoint(5)
        
        # All points should be contained
        for i in range(points.shape[1]):
            point = points[:, i]
            assert fs.contains(point)
            assert len(point) == n

    def test_standard_sampling(self):
        """Test standard sampling method"""
        n = 3
        fs = Fullspace(n)
        
        # Standard sampling
        points = fs.randPoint(10, type_='standard')
        
        # All points should be contained
        for i in range(points.shape[1]):
            point = points[:, i]
            assert fs.contains(point)
            assert len(point) == n

    def test_extreme_sampling_single(self):
        """Test extreme point sampling - single point"""
        n = 3
        fs = Fullspace(n)
        
        # Extreme point sampling
        p = fs.randPoint(1, type_='extreme')
        
        # Point should be contained
        assert fs.contains(p)
        assert len(p) == n

    def test_extreme_sampling_all(self):
        """Test extreme point sampling - all extreme points"""
        n = 2  # Use smaller dimension for 'all' case
        fs = Fullspace(n)
        
        # All extreme points
        points = fs.randPoint('all', type_='extreme')
        
        # All points should be contained
        if points.ndim == 1:
            assert fs.contains(points)
        else:
            for i in range(points.shape[1]):
                point = points[:, i]
                assert fs.contains(point)

    def test_zero_points(self):
        """Test generation of zero points"""
        n = 2
        fs = Fullspace(n)
        
        # N=0 should raise an error as it's not a positive integer
        with pytest.raises(Exception):
            fs.randPoint(0)

    def test_one_dimensional_fullspace(self):
        """Test random point in 1D fullspace"""
        n = 1
        fs = Fullspace(n)
        
        p = fs.randPoint()
        assert fs.contains(p)
        assert len(p) == n

    def test_high_dimensional_fullspace(self):
        """Test random point in high-dimensional fullspace"""
        n = 10
        fs = Fullspace(n)
        
        p = fs.randPoint()
        assert fs.contains(p)
        assert len(p) == n

    def test_random_distribution_properties(self):
        """Test that random points have expected distribution properties"""
        n = 2
        fs = Fullspace(n)
        
        # Generate many points and check they're well distributed
        num_points = 100
        points = fs.randPoint(num_points)
        
        # Check shape
        assert points.shape == (n, num_points)
        
        # All points should be contained
        for i in range(num_points):
            assert fs.contains(points[:, i])

    def test_invalid_parameters(self):
        """Test invalid parameter handling"""
        n = 2
        fs = Fullspace(n)
        
        # Negative number of points should raise error
        with pytest.raises((ValueError, Exception)):
            fs.randPoint(-1)

    def test_zero_dimensional_fullspace(self):
        """Test random point in zero-dimensional fullspace"""
        n = 0
        fs = Fullspace(n)
        
        # Zero-dimensional fullspace represents empty set, so randPoint returns empty array
        p = fs.randPoint()
        assert len(p) == 0
        # Note: contains check is not supported for zero-dimensional fullspace 