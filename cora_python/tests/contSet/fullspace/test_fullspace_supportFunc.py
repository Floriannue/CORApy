"""
test_supportFunc - unit test function for fullspace supportFunc

Tests the support function functionality for fullspace objects.

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       05-April-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace.fullspace import Fullspace
from cora_python.contSet.interval.interval import Interval


class TestFullspaceSupportFunc:
    """Test class for fullspace supportFunc method"""

    def test_support_function_upper(self):
        """Test support function in upper direction"""
        n = 2
        fs = Fullspace(n)
        
        # Direction
        dir = np.array([1, 0.5])
        
        # Compute support function (upper by default)
        val, x = fs.supportFunc(dir)
        
        # For fullspace, support function should be infinite
        assert val == np.inf
        assert np.all(x == np.array([np.inf, np.inf]))

    def test_support_function_lower(self):
        """Test support function in lower direction"""
        n = 2
        fs = Fullspace(n)
        
        # Direction
        dir = np.array([1, 0.5])
        
        # Compute support function (lower direction)
        val, x = fs.supportFunc(dir, type='lower')
        
        # For fullspace, lower support function should be negative infinite
        assert val == -np.inf
        assert np.all(x == np.array([-np.inf, -np.inf]))

    def test_support_function_range(self):
        """Test support function range"""
        n = 2
        fs = Fullspace(n)
        
        # Direction
        dir = np.array([1, 0.5])
        
        # Compute support function range
        val, x = fs.supportFunc(dir, type='range')
        
        # For fullspace, range should be [-Inf, Inf]
        expected_interval = Interval(np.array([-np.inf]), np.array([np.inf]))
        assert val.isequal(expected_interval)
        
        # Support points should be infinite bounds
        expected_x = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        assert np.all(x == expected_x)

    def test_support_function_zero_direction(self):
        """Test support function with zero direction"""
        n = 2
        fs = Fullspace(n)
        
        # Zero direction
        dir = np.array([0, 0])
        
        # Support function should still be infinite
        val, x = fs.supportFunc(dir)
        assert val == np.inf
        assert np.all(x == np.array([np.inf, np.inf]))

    def test_support_function_negative_direction(self):
        """Test support function with negative direction"""
        n = 2
        fs = Fullspace(n)
        
        # Negative direction
        dir = np.array([-1, -0.5])
        
        # Support function should still be infinite
        val, x = fs.supportFunc(dir)
        assert val == np.inf
        assert np.all(x == np.array([np.inf, np.inf]))

    def test_support_function_unit_directions(self):
        """Test support function with unit coordinate directions"""
        n = 3
        fs = Fullspace(n)
        
        # Test each coordinate direction
        for i in range(n):
            dir = np.zeros(n)
            dir[i] = 1
            
            val, x = fs.supportFunc(dir)
            assert val == np.inf
            assert np.all(x == np.inf * np.ones(n))

    def test_support_function_high_dimension(self):
        """Test support function in high dimensions"""
        n = 10
        fs = Fullspace(n)
        
        # Random direction
        dir = np.random.randn(n)
        dir = dir / np.linalg.norm(dir)  # Normalize
        
        val, x = fs.supportFunc(dir)
        assert val == np.inf
        assert np.all(x == np.inf * np.ones(n))

    def test_support_function_one_dimension(self):
        """Test support function in 1D"""
        n = 1
        fs = Fullspace(n)
        
        # Direction
        dir = np.array([1])
        
        val, x = fs.supportFunc(dir)
        assert val == np.inf
        assert np.all(x == np.array([np.inf]))

    def test_invalid_direction_dimension(self):
        """Test support function with wrong direction dimension"""
        n = 2
        fs = Fullspace(n)
        
        # Wrong dimension direction
        dir = np.array([1, 0, 1])  # 3D direction for 2D fullspace
        
        with pytest.raises((ValueError, Exception)):
            fs.supportFunc(dir)

    def test_support_function_all_types(self):
        """Test all support function types consistently"""
        n = 2
        fs = Fullspace(n)
        dir = np.array([1, 1])
        
        # Test upper
        val_upper, _ = fs.supportFunc(dir, type='upper')
        assert val_upper == np.inf
        
        # Test lower  
        val_lower, _ = fs.supportFunc(dir, type='lower')
        assert val_lower == -np.inf
        
        # Test range
        val_range, _ = fs.supportFunc(dir, type='range')
        expected = Interval(np.array([-np.inf]), np.array([np.inf]))
        assert val_range.isequal(expected) 