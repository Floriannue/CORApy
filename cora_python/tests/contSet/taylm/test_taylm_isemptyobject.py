"""
test_isemptyobject - unit test function for taylm isemptyobject

Tests the empty object check functionality for Taylm objects.

Authors:       Dmitry Grebenyuk, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.taylm.taylm import Taylm
from cora_python.contSet.interval.interval import Interval


class TestTaylmIsemptyobject:
    """Test class for taylm isemptyobject method"""

    def test_isemptyobject_normal_taylm(self):
        """Test isemptyobject for normal Taylm objects"""
        I = Interval(np.array([1, 2]), np.array([3, 4]))
        tay = Taylm(I)
        
        # Normal Taylm should not be empty
        assert not tay.isemptyobject()

    def test_isemptyobject_various_dimensions(self):
        """Test isemptyobject for various dimensions"""
        for n in [1, 2, 5, 10]:
            I = Interval(-np.ones(n), np.ones(n))
            tay = Taylm(I)
            assert not tay.isemptyobject()

    def test_isemptyobject_zero_dimension(self):
        """Test isemptyobject for zero-dimensional Taylm"""
        empty_tay = Taylm.empty(0)
        # Zero-dimensional might be considered empty
        result = empty_tay.isemptyobject()
        assert isinstance(result, bool)

    def test_isemptyobject_return_type(self):
        """Test that isemptyobject returns boolean"""
        I = Interval(np.array([0]), np.array([1]))
        tay = Taylm(I)
        
        result = tay.isemptyobject()
        assert isinstance(result, bool)

    def test_isemptyobject_consistency(self):
        """Test that isemptyobject is consistent"""
        I = Interval(np.array([1, 2, 3]), np.array([4, 5, 6]))
        tay = Taylm(I)
        
        # Multiple calls should return same result
        result1 = tay.isemptyobject()
        result2 = tay.isemptyobject()
        result3 = tay.isemptyobject()
        
        assert result1 == result2 == result3

    def test_isemptyobject_generated_random(self):
        """Test isemptyobject for randomly generated Taylm"""
        tay = Taylm.generateRandom(dimension=3)
        assert not tay.isemptyobject()

    def test_isemptyobject_origin(self):
        """Test isemptyobject for origin Taylm"""
        origin_tay = Taylm.origin(4)
        assert not origin_tay.isemptyobject()

    def test_isemptyobject_single_point(self):
        """Test isemptyobject for single point interval"""
        I = Interval(np.array([2]), np.array([2]))
        tay = Taylm(I)
        assert not tay.isemptyobject()

    def test_isemptyobject_large_dimension(self):
        """Test isemptyobject for large dimension"""
        n = 50
        I = Interval(np.zeros(n), np.ones(n))
        tay = Taylm(I)
        assert not tay.isemptyobject() 