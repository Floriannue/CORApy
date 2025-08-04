"""
test_zonotope_contains_ - unit test function of contains_

Tests the contains_ method for zonotope objects to check containment.
Direct translation from MATLAB test_zonotope_contains.m

Syntax:
    pytest cora_python/tests/contSet/zonotope/test_zonotope_contains_.py

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 12-July-2023 (MATLAB)
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope

# 工具函数，兼容 bool 和 numpy 数组

def assert_bool(res, expected):
    if isinstance(res, np.ndarray):
        if expected:
            assert np.all(res)
        else:
            assert not np.any(res)
    else:
        assert res == expected


class TestZonotopeContains:
    def test_point_in_zono(self):
        """Test point-in-zonotope containment"""
        # 2D zonotopes
        c = np.array([[0], [1]])
        G = np.array([[1, 2, 1], [-1, 0, 1]])
        Z = Zonotope(c, G)
        
        # Point inside
        point_in = np.array([[1.5797], [1.3568]])
        res, cert, scaling = Z.contains_(point_in)
        assert_bool(res, True)
        assert cert == True
        
        # Point outside
        point_out = np.array([[3.5797], [2.3568]])
        res, cert, scaling = Z.contains_(point_out)
        assert_bool(res, False)
        assert cert == True

    def test_degenerate_zonotope(self):
        """Test degenerate zonotope containment"""
        Z = Zonotope(np.array([[-1], [0]]), np.array([[1], [1]]))
        
        # Point inside
        point_in = np.array([[-0.5], [0.5]])
        res, cert, scaling = Z.contains_(point_in)
        assert_bool(res, True)
        assert cert == True
        
        # Point outside
        point_out = np.array([[-0.5], [-0.5]])
        res, cert, scaling = Z.contains_(point_out)
        assert_bool(res, False)
        assert cert == True

    def test_almost_degenerate_zonotope(self):
        """Test almost degenerate zonotope containment"""
        Z = Zonotope(np.array([[-1], [0]]), np.array([[1, 1e-5], [1, 2e-5]]))
        
        # Point inside
        point_in = np.array([[-0.5], [0.5]])
        res, cert, scaling = Z.contains_(point_in)
        assert_bool(res, True)
        assert cert == True
        
        # Point outside
        point_out = np.array([[-0.5], [-0.5]])
        res, cert, scaling = Z.contains_(point_out)
        assert_bool(res, False)
        assert cert == True

    def test_outer_zonotope_is_interval(self):
        """Test when outer zonotope is an interval"""
        from cora_python.contSet.interval.interval import Interval
        Z1 = Zonotope(Interval(np.array([1,2]), np.array([4,6])))
        Z2 = Zonotope(np.array([[3],[4]]), np.array([[1],[1]]))
        res, cert, scaling = Z1.contains_(Z2)
        assert_bool(res, True)

    def test_zono_in_zono(self):
        """Test zonotope-in-zonotope containment (both directions)"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2, 1], [-1, 0, 1]])
        Z1 = Zonotope(c, G)
        c = np.array([[-1], [1.5]])
        G = np.array([[0.2, 0], [-0.1, 0.1]])
        Z2 = Zonotope(c, G)
        res, cert, scaling = Z1.contains_(Z2)
        assert_bool(res, True)
        res, cert, scaling = Z2.contains_(Z1)
        assert_bool(res, False)

    def test_inner_zonotope_is_point(self):
        """Test when inner zonotope is just a point"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2, 1], [-1, 0, 1]])
        Z1 = Zonotope(c, G)
        
        c = np.array([[-1], [1.5]])
        Z2 = Zonotope(c)  # Point zonotope
        
        res, cert, scaling = Z1.contains_(Z2)
        # This should work as point containment
        assert cert == True

    def test_approx_st_method(self):
        """Test approximate ST method"""
        c = np.array([[0], [1]])
        G = np.array([[1, 2, 1], [-1, 0, 1]])
        Z1 = Zonotope(c, G)
        
        c = np.array([[-1], [1.5]])
        Z2 = Zonotope(c)  # Point zonotope
        
        # Choose LP method for containment
        res, cert, scaling = Z1.contains_(Z2, method='approx:st')
        assert cert == True

    def test_both_zonotopes_are_points(self):
        """Test when both zonotopes are just points"""
        Z1 = Zonotope(np.zeros((4, 1)))
        Z2 = Zonotope(np.ones((4, 1)))
        
        # Same point contains itself
        res, cert, scaling = Z1.contains_(Z1)
        assert_bool(res, True)
        assert cert == True
        
        # Different points
        res, cert, scaling = Z1.contains_(Z2)
        assert_bool(res, False)
        assert cert == True

    def test_degenerate_sets(self):
        """Test two sets with one being degenerate"""
        c = np.array([[5.000], [0.000]])
        G = np.array([[0.000], [1.000]])
        Z1 = Zonotope(c, G)
        
        c = np.array([[5.650], [0.000]])
        G = np.array([[0.000, 0.050, 0.000, 0.000, 0.000], 
                     [0.937, 0.000, -0.005, -0.000, 0.000]])
        Z2 = Zonotope(c, G)
        
        res, cert, scaling = Z1.contains_(Z2)
        assert_bool(res, False)
        assert cert == True

    def test_empty_zonotope(self):
        """Test empty zonotope containment"""
        Z = Zonotope.empty(2)
        res, cert, scaling = Z.contains_(Z)
        assert_bool(res, True)

if __name__ == "__main__":
    pytest.main([__file__]) 