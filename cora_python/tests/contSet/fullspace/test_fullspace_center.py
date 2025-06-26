"""
test_center - unit test function for fullspace center

Tests the center computation functionality for fullspace objects.

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       05-April-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace.fullspace import Fullspace


class TestFullspaceCenter:
    """Test class for fullspace center method"""

    def test_basic_center(self):
        """Test basic center computation"""
        n = 2
        fs = Fullspace(n)
        
        # Compute center
        c = fs.center()
        c_true = np.zeros(n)
        
        # Compare results
        assert np.allclose(c, c_true)

    def test_center_different_dimensions(self):
        """Test center computation for different dimensions"""
        for n in [1, 3, 5, 10]:
            fs = Fullspace(n)
            c = fs.center()
            
            # Center should always be zero vector
            assert np.allclose(c, np.zeros(n))
            assert len(c) == n

    def test_center_zero_dimension(self):
        """Test center computation for zero-dimensional fullspace"""
        n = 0
        fs = Fullspace(n)
        
        c = fs.center()
        assert len(c) == 0
        assert np.allclose(c, np.zeros(0))

    def test_center_properties(self):
        """Test properties of the center"""
        n = 3
        fs = Fullspace(n)
        
        c = fs.center()
        
        # Center should be contained in the fullspace
        assert fs.contains(c)
        
        # Center should have correct shape
        assert c.shape == (n,)
        
        # Center should be finite (not infinite)
        assert np.all(np.isfinite(c))

    def test_center_consistency(self):
        """Test that center computation is consistent"""
        n = 4
        fs = Fullspace(n)
        
        # Compute center multiple times
        c1 = fs.center()
        c2 = fs.center()
        c3 = fs.center()
        
        # Should always return the same result
        assert np.allclose(c1, c2)
        assert np.allclose(c2, c3)

    def test_center_type(self):
        """Test that center returns correct type"""
        n = 2
        fs = Fullspace(n)
        
        c = fs.center()
        
        # Should return numpy array
        assert isinstance(c, np.ndarray)
        assert c.dtype == np.float64

    def test_center_high_dimension(self):
        """Test center computation in high dimensions"""
        n = 100
        fs = Fullspace(n)
        
        c = fs.center()
        
        # Should still be zero vector
        assert np.allclose(c, np.zeros(n))
        assert len(c) == n

    def test_center_one_dimension(self):
        """Test center computation in 1D"""
        n = 1
        fs = Fullspace(n)
        
        c = fs.center()
        
        # Should be [0]
        assert np.allclose(c, np.array([0]))
        assert len(c) == 1 