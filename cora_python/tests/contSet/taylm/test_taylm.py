"""
test_taylm - unit test function for Taylm constructor

Tests the Taylor model constructor functionality.

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


class TestTaylm:
    """Test class for Taylm constructor"""

    def test_constructor_scalar_interval(self):
        """Test constructor with scalar interval"""
        I = Interval(np.array([1]), np.array([2]))
        tay = Taylm(I)
        
        # Should create a valid Taylm object
        assert isinstance(tay, Taylm)
        assert tay.dim() == 1

    def test_constructor_multidimensional_interval(self):
        """Test constructor with multidimensional interval"""
        lb = np.array([-3, -2, -5])
        ub = np.array([4, 2, 1])
        I = Interval(lb, ub)
        tay = Taylm(I)
        
        # Should create a valid Taylm object
        assert isinstance(tay, Taylm)
        assert tay.dim() == 3

    def test_constructor_validation(self):
        """Test constructor input validation"""
        # Valid interval
        I = Interval(np.array([0]), np.array([1]))
        tay = Taylm(I)
        assert isinstance(tay, Taylm)

    def test_constructor_properties(self):
        """Test constructor sets proper properties"""
        I = Interval(np.array([-1, 0]), np.array([1, 2]))
        tay = Taylm(I)
        
        # Check basic properties
        assert tay.dim() == 2
        assert not tay.isemptyobject()

    def test_constructor_edge_cases(self):
        """Test constructor with edge cases"""
        # Single point interval
        I = Interval(np.array([5]), np.array([5]))
        tay = Taylm(I)
        assert tay.dim() == 1

    def test_constructor_zero_width_interval(self):
        """Test constructor with zero-width interval"""
        I = Interval(np.array([2, 3]), np.array([2, 3]))
        tay = Taylm(I)
        assert tay.dim() == 2

    def test_invalid_constructor_inputs(self):
        """Test constructor with invalid inputs"""
        # Invalid interval (inf > sup)
        with pytest.raises((ValueError, Exception)):
            I = Interval(np.array([2]), np.array([1]))
            Taylm(I) 