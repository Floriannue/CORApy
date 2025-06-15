"""
test_interval_contains_ - unit test function of contains_

Tests the contains_ method for interval objects to check containment.
Direct translation from MATLAB test_interval_contains.m

Syntax:
    pytest cora_python/tests/contSet/interval/test_interval_contains_.py

Authors: Mark Wetzlinger, Adrian Kulmburg (MATLAB)
         Python translation by AI Assistant
Written: 27-September-2019 (MATLAB)
Last update: 12-March-2021 (MW, add empty case)
             01-July-2021 (AK, integrated test_interval_containsPoint)
             03-December-2023 (MW, add unbounded case)
"""

import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalContains:
    def test_empty_case(self):
        """Test empty interval containment"""
        I = Interval.empty(2)
        res, cert, scaling = I.contains_(I)
        assert res == True
        assert cert == True
        assert scaling == 0

    def test_zonotope_in_interval(self):
        """Test zonotope-in-interval containment"""
        I = Interval(np.array([[-3], [-2]]), np.array([[5], [4]]))
        
        # Create a simple zonotope-like test with vertices
        # For now, test with points that represent zonotope vertices
        Z_in_vertices = np.array([[0.5, -1.5, 2.5, -0.5], [0, -1.7, 0.7, 1.7]])
        res, cert, scaling = I.contains_(Z_in_vertices)
        assert np.all(res)
        assert np.all(cert)
        
        Z_out_vertices = np.array([[6.5, 4.5, 8.5, 5.5], [-3, -4.7, -2.3, -1.7]])
        res, cert, scaling = I.contains_(Z_out_vertices)
        assert not np.all(res)
        assert np.all(cert)

    def test_point_containment_case(self):
        """Test point containment"""
        I = Interval(np.array([[-3], [-9], [-4], [-7], [-1]]), 
                    np.array([[4], [2], [6], [3], [8]]))
        
        p_inside = np.array([[0, 1, -2, -3], [0, -4, -6, 2], [0, 3, -2, 6], [0, -6, 2, -7], [0, 5, 7, 8]])
        res, cert, scaling = I.contains_(p_inside)
        assert len(res) == 4
        assert np.all(res)
        assert np.all(cert)
        
        p_outside = np.array([[5, 1, -2, -3], [3, -4, -5, 2], [7, 3, -2, 6], [4, -6, 6, -7], [9, 5, 7, 10]])
        res, cert, scaling = I.contains_(p_outside)
        assert not np.all(res)
        assert np.all(cert)

    def test_unbounded_intervals(self):
        """Test unbounded intervals"""
        I1 = Interval(np.array([[-np.inf]]), np.array([[np.inf]]))
        I2 = Interval(np.array([[-np.inf]]), np.array([[0]]))
        res, cert, scaling = I1.contains_(I2)
        assert res == True
        assert cert == True
        
        I2 = Interval(np.array([[0]]), np.array([[np.inf]]))
        res, cert, scaling = I1.contains_(I2)
        assert res == True
        assert cert == True
        
        p = np.array([[-np.inf, 0, np.inf]])
        res, cert, scaling = I1.contains_(p)
        assert np.all(res)
        assert np.all(cert)

    def test_nd_arrays(self):
        """Test n-d arrays"""
        # Create multi-dimensional bounds
        lb = np.zeros((2, 2, 1, 3, 2))
        lb[:, :, 0, 0, 0] = [[1, 2], [3, 5]]
        lb[:, :, 0, 1, 0] = [[0, -1], [-2, 3]]
        lb[:, :, 0, 2, 0] = [[1, 1], [-1, 0]]
        lb[:, :, 0, 0, 1] = [[-3, 2], [0, 1]]
        
        ub = np.zeros((2, 2, 1, 3, 2))
        ub[:, :, 0, 0, 0] = [[1.5, 4], [4, 10]]
        ub[:, :, 0, 1, 0] = [[1, 2], [0, 4]]
        ub[:, :, 0, 2, 0] = [[2, 3], [-0.5, 2]]
        ub[:, :, 0, 0, 1] = [[-1, 3], [0, 2]]
        
        I = Interval(lb, ub)
        c = (lb + ub) / 2
        
        # Test with center points
        test_points = np.concatenate([c, c], axis=4)
        res, cert, scaling = I.contains_(test_points)
        assert res.shape == (1, 2)
        assert np.all(res)

    def test_contains_with_scaling(self):
        """Test containment with scaling computation"""
        I1 = Interval(np.array([[-1], [-2]]), np.array([[2], [3]]))
        I2 = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        
        res, cert, scaling = I1.contains_(I2, scalingToggle=True)
        assert res == True
        assert cert == True
        assert isinstance(scaling, (int, float))
        assert scaling <= 1.0

    def test_contains_with_tolerance(self):
        """Test containment with tolerance"""
        I = Interval(np.array([[0]]), np.array([[1]]))
        point = np.array([[1.001]])
        
        # Without tolerance
        res, cert, scaling = I.contains_(point, tol=0)
        assert res == False
        assert cert == True
        
        # With tolerance
        res, cert, scaling = I.contains_(point, tol=0.01)
        assert res == True
        assert cert == True

    def test_interval_in_interval_containment(self):
        """Test interval-in-interval containment"""
        I1 = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        I2 = Interval(np.array([[-1], [2]]), np.array([[2], [3]]))
        
        res, cert, scaling = I1.contains_(I2)
        assert res == True
        assert cert == True
        
        # Test non-containment
        I3 = Interval(np.array([[-3], [2]]), np.array([[2], [5]]))
        res, cert, scaling = I1.contains_(I3)
        assert res == False
        assert cert == True

    def test_empty_interval_containment(self):
        """Test empty interval cases"""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        I_empty = Interval.empty(2)
        
        # Non-empty contains empty
        res, cert, scaling = I.contains_(I_empty)
        assert res == True
        assert cert == True
        assert scaling == 0
        
        # Empty contains non-empty
        res, cert, scaling = I_empty.contains_(I)
        assert res == False
        assert cert == True
        assert scaling == np.inf

    def test_point_array_containment(self):
        """Test containment of point arrays"""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        
        # Multiple points
        points = np.array([[0, 5, -1], [2, 2, 3]])
        res, cert, scaling = I.contains_(points)
        
        expected_res = np.array([True, False, True])
        np.testing.assert_array_equal(res, expected_res)
        assert np.all(cert)


if __name__ == "__main__":
    pytest.main([__file__]) 