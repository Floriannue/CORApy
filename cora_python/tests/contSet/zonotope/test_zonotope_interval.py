"""
test_zonotope_interval - unit test function of interval

Syntax:
    python -m pytest test_zonotope_interval.py

Inputs:
    -

Outputs:
    test results

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 26-July-2016 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeInterval:
    """Test class for zonotope interval method"""
    
    def test_basic_interval_conversion(self):
        """Test basic conversion to interval"""
        # Create zonotope
        Z1 = Zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        
        # Create interval
        I1 = Z1.interval()
        
        # Obtain results
        lb = I1.infimum()
        ub = I1.supremum()
        
        # True results
        true_lb = np.array([-10, -8])
        true_ub = np.array([2, 10])
        
        # Check result
        np.testing.assert_array_almost_equal(lb, true_lb)
        np.testing.assert_array_almost_equal(ub, true_ub)
    
    def test_1d_interval(self):
        """Test 1D zonotope to interval conversion"""
        Z = Zonotope(np.array([0]), np.array([[1, 2, 0.5]]))
        I = Z.interval()
        
        lb = I.infimum()
        ub = I.supremum()
        
        # Should be [-3.5, 3.5]
        expected_lb = np.array([-3.5])
        expected_ub = np.array([3.5])
        
        np.testing.assert_array_almost_equal(lb, expected_lb)
        np.testing.assert_array_almost_equal(ub, expected_ub)
    
    def test_origin_interval(self):
        """Test origin zonotope to interval"""
        Z = Zonotope.origin(2)
        I = Z.interval()
        
        lb = I.infimum()
        ub = I.supremum()
        
        # Should be [0, 0] for both bounds
        np.testing.assert_array_almost_equal(lb, np.zeros(2))
        np.testing.assert_array_almost_equal(ub, np.zeros(2))
    
    def test_box_zonotope_interval(self):
        """Test axis-aligned box (diagonal generator matrix)"""
        c = np.array([1, 2])
        G = np.array([[2, 0], [0, 3]])  # Axis-aligned box
        Z = Zonotope(c, G)
        I = Z.interval()
        
        lb = I.infimum()
        ub = I.supremum()
        
        expected_lb = np.array([-1, -1])  # [1-2, 2-3]
        expected_ub = np.array([3, 5])    # [1+2, 2+3]
        
        np.testing.assert_array_almost_equal(lb, expected_lb)
        np.testing.assert_array_almost_equal(ub, expected_ub)
    
    def test_empty_zonotope_interval(self):
        """Test empty zonotope to interval"""
        Z_empty = Zonotope.empty(2)
        I_empty = Z_empty.interval()
        
        # Should be empty interval
        assert I_empty.isemptyobject()
    
    def test_single_point_interval(self):
        """Test zonotope representing single point"""
        c = np.array([5, -2])
        Z = Zonotope(c)  # No generators
        I = Z.interval()
        
        lb = I.infimum()
        ub = I.supremum()
        
        # Both bounds should equal the center
        np.testing.assert_array_almost_equal(lb, c)
        np.testing.assert_array_almost_equal(ub, c)
    
    def test_interval_containment(self):
        """Test that interval contains the original zonotope"""
        Z = Zonotope(np.array([0, 0]), np.array([[1, 2], [3, 1]]))
        I = Z.interval()
        
        # Sample some points from the zonotope
        points = Z.randPoint_(10)
        
        # All points should be contained in the interval
        for i in range(points.shape[1]):
            point = points[:, i]
            assert I.contains_(point)


if __name__ == "__main__":
    test_instance = TestZonotopeInterval()
    
    # Run all tests
    test_instance.test_basic_interval_conversion()
    test_instance.test_1d_interval()
    test_instance.test_origin_interval()
    test_instance.test_box_zonotope_interval()
    test_instance.test_empty_zonotope_interval()
    test_instance.test_single_point_interval()
    test_instance.test_interval_containment()
    
    print("All zonotope interval tests passed!") 