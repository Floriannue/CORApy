"""
test_origin - unit test function for taylm origin

Tests the origin creation functionality for Taylm objects.

Authors:       Dmitry Grebenyuk, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.taylm.taylm import Taylm


class TestTaylmOrigin:
    """Test class for taylm origin method"""

    def test_origin_basic(self):
        """Test basic origin creation"""
        n = 2
        origin_tay = Taylm.origin(n)
        
        # Should create a valid Taylm object
        assert isinstance(origin_tay, Taylm)
        assert origin_tay.dim() == n

    def test_origin_various_dimensions(self):
        """Test origin creation for various dimensions"""
        for n in [1, 3, 5, 10]:
            origin_tay = Taylm.origin(n)
            assert isinstance(origin_tay, Taylm)
            assert origin_tay.dim() == n

    def test_origin_zero_dimension(self):
        """Test origin creation for zero dimension"""
        origin_tay = Taylm.origin(0)
        assert isinstance(origin_tay, Taylm)
        assert origin_tay.dim() == 0

    def test_origin_static_method(self):
        """Test that origin is a static method"""
        # Should be callable without instance
        origin_tay = Taylm.origin(3)
        assert isinstance(origin_tay, Taylm)

    def test_origin_invalid_dimension(self):
        """Test origin creation with invalid dimension"""
        # Negative dimension should raise error
        with pytest.raises((ValueError, Exception)):
            Taylm.origin(-1)

    def test_origin_large_dimension(self):
        """Test origin creation for large dimension"""
        n = 100
        origin_tay = Taylm.origin(n)
        assert isinstance(origin_tay, Taylm)
        assert origin_tay.dim() == n

    def test_origin_consistency(self):
        """Test that origin creation is consistent"""
        n = 4
        origin1 = Taylm.origin(n)
        origin2 = Taylm.origin(n)
        
        # Both should be valid Taylm objects with same dimension
        assert isinstance(origin1, Taylm)
        assert isinstance(origin2, Taylm)
        assert origin1.dim() == origin2.dim() == n

    def test_origin_not_empty(self):
        """Test that origin is not an empty object"""
        n = 3
        origin_tay = Taylm.origin(n)
        assert not origin_tay.isemptyobject() 