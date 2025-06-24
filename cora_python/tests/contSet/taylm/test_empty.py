"""
test_empty - unit test function for taylm empty

Tests the empty set creation functionality for Taylm objects.

Authors:       Dmitry Grebenyuk, Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.taylm.taylm import Taylm


class TestTaylmEmpty:
    """Test class for taylm empty method"""

    def test_empty_basic(self):
        """Test basic empty creation"""
        n = 2
        empty_tay = Taylm.empty(n)
        
        # Should create a valid Taylm object
        assert isinstance(empty_tay, Taylm)
        assert empty_tay.dim() == n

    def test_empty_various_dimensions(self):
        """Test empty creation for various dimensions"""
        for n in [1, 3, 5, 10]:
            empty_tay = Taylm.empty(n)
            assert isinstance(empty_tay, Taylm)
            assert empty_tay.dim() == n

    def test_empty_zero_dimension(self):
        """Test empty creation for zero dimension"""
        empty_tay = Taylm.empty(0)
        assert isinstance(empty_tay, Taylm)
        assert empty_tay.dim() == 0

    def test_empty_static_method(self):
        """Test that empty is a static method"""
        # Should be callable without instance
        empty_tay = Taylm.empty(3)
        assert isinstance(empty_tay, Taylm)

    def test_empty_invalid_dimension(self):
        """Test empty creation with invalid dimension"""
        # Negative dimension should raise error
        with pytest.raises((ValueError, Exception)):
            Taylm.empty(-1)

    def test_empty_large_dimension(self):
        """Test empty creation for large dimension"""
        n = 100
        empty_tay = Taylm.empty(n)
        assert isinstance(empty_tay, Taylm)
        assert empty_tay.dim() == n

    def test_empty_consistency(self):
        """Test that empty creation is consistent"""
        n = 4
        empty1 = Taylm.empty(n)
        empty2 = Taylm.empty(n)
        
        # Both should be valid Taylm objects with same dimension
        assert isinstance(empty1, Taylm)
        assert isinstance(empty2, Taylm)
        assert empty1.dim() == empty2.dim() == n 