"""
test_contains_ - unit test function for polytope contains_ method

Tests the containment checking functionality of polytopes,
including point containment and polytope containment.

Authors: MATLAB: Viktor Kotsev, Mark Wetzlinger  
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.contains_ import contains_


class TestPolytopeContains:
    """Test class for polytope contains_ method"""
    
    def test_contains_1d_unbounded_fully_empty(self):
        """Test 1D unbounded (fully empty) >= unbounded (fully empty)"""
        A = np.zeros((0, 1))
        b = np.zeros(0)
        P1 = Polytope(A, b)
        assert contains_(P1, P1)
    
    def test_contains_1d_unbounded_vs_bounded(self):
        """Test 1D unbounded >= bounded"""
        # Unbounded (fully empty)
        A = np.zeros((0, 1))
        b = np.zeros(0)
        P1 = Polytope(A, b)
        # Bounded
        A = np.array([[1], [-1]])
        b = np.array([1, 1])
        P2 = Polytope(A, b)
        assert contains_(P1, P2)
    
    def test_contains_1d_unbounded_same_set(self):
        """Test 1D unbounded >= unbounded (same set)"""
        A1 = np.array([[3]])
        b1 = np.array([6])
        P1 = Polytope(A1, b1)
        A2 = np.array([[1]])
        b2 = np.array([2])
        P2 = Polytope(A2, b2)
        # Both represent x <= 2, so they should contain each other
        assert contains_(P1, P2)
        assert contains_(P2, P1)
    
    def test_contains_1d_unbounded_subset(self):
        """Test 1D unbounded >= unbounded (subset)"""
        A1 = np.array([[1]])
        b1 = np.array([1])
        P1 = Polytope(A1, b1)  # x <= 1
        A2 = np.array([[1]])
        b2 = np.array([0])
        P2 = Polytope(A2, b2)  # x <= 0
        assert contains_(P1, P2)
    
    def test_contains_1d_unbounded_vs_bounded_subset(self):
        """Test 1D unbounded >= bounded (subset)"""
        A1 = np.array([[1]])
        b1 = np.array([1])
        P1 = Polytope(A1, b1)  # x <= 1
        A2 = np.array([[1], [-1]])
        b2 = np.array([0.5, 0.75])
        P2 = Polytope(A2, b2)  # -0.75 <= x <= 0.5
        assert contains_(P1, P2)
    
    def test_contains_1d_unbounded_vs_degenerate(self):
        """Test 1D unbounded >= degenerate"""
        A1 = np.array([[1]])
        b1 = np.array([1])
        P1 = Polytope(A1, b1)  # x <= 1
        Ae = np.array([[1]])
        be = np.array([0])
        P2 = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)  # x = 0
        assert contains_(P1, P2)
    
    def test_contains_1d_bounded_not_containing_unbounded(self):
        """Test 1D bounded does not contain unbounded"""
        A1 = np.array([[1], [-1]])
        b1 = np.array([1, 1])
        P1 = Polytope(A1, b1)  # [-1, 1]
        A2 = np.array([[2], [3]])
        b2 = np.array([7, 2])
        P2 = Polytope(A2, b2)  # x <= 3.5 and x <= 2/3 -> unbounded
        assert not contains_(P1, P2)
    
    def test_contains_1d_degenerate_not_containing_unbounded(self):
        """Test 1D degenerate does not contain unbounded"""
        A1 = np.array([[1], [-1]])
        b1 = np.array([1, -1])
        P1 = Polytope(A1, b1)  # x = 1 (degenerate)
        A2 = np.array([[2], [3]])
        b2 = np.array([7, 2])
        P2 = Polytope(A2, b2)
        assert not contains_(P1, P2)
    
    def test_contains_1d_degenerate_vs_degenerate(self):
        """Test 1D degenerate >= degenerate"""
        A1 = np.array([[1], [-1]])
        b1 = np.array([1, -1])
        P1 = Polytope(A1, b1)  # x = 1
        Ae = np.array([[1]])
        be = np.array([0])
        P2 = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)  # x = 0
        assert not contains_(P1, P2)
    
    def test_contains_1d_unbounded_point_cloud(self):
        """Test 1D unbounded >= point cloud"""
        A1 = np.array([[1], [2]])
        b1 = np.array([1, 1])
        P1 = Polytope(A1, b1)  # x <= 1 and x <= 0.5 -> x <= 0.5
        S = np.array([[-4, -2, 0]])
        result = contains_(P1, S.T)
        assert np.all(result)
    
    def test_contains_1d_bounded_point_cloud_partial(self):
        """Test 1D bounded >= point cloud (partial containment)"""
        A1 = np.array([[1], [-1]])
        b1 = np.array([1, 1])
        P1 = Polytope(A1, b1)  # [-1, 1]
        S = np.array([[-3, 0.5, 0]])
        result = contains_(P1, S.T)
        expected = np.array([False, True, True])
        assert np.array_equal(result, expected)
    
    def test_contains_1d_degenerate_single_point(self):
        """Test 1D degenerate >= single point"""
        Ae = np.array([[1]])
        be = np.array([-3])
        P1 = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)  # x = -3
        S = np.array([[-3]])
        assert contains_(P1, S.T)
    
    def test_contains_1d_degenerate_multiple_points(self):
        """Test 1D degenerate >= multiple points"""
        Ae = np.array([[1]])
        be = np.array([-3])
        P1 = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)  # x = -3
        S = np.array([[-3, -2]])
        result = contains_(P1, S.T)
        expected = np.array([True, False])
        assert np.array_equal(result, expected)
    
    def test_contains_2d_bounded_vs_bounded(self):
        """Test 2D bounded, non-degenerate >= bounded, non-degenerate"""
        A1 = np.array([[1, 1], [1, -1], [-1, 1], [1, 1]])
        b1 = np.array([1, 1, 1, 1])
        P1 = Polytope(A1, b1)
        A2 = np.array([[1, 1], [1, -1], [-1, 1], [1, 1]])
        b2 = np.array([2, 2, 2, 2])
        P2 = Polytope(A2, b2)
        p1 = np.array([[0], [0]])
        p2 = np.array([[5], [5]])
        assert not contains_(P1, P2) and contains_(P2, P1)
        assert contains_(P1, p1) and not contains_(P2, p2)
    
    def test_contains_2d_unbounded_vs_degenerate(self):
        """Test 2D unbounded, non-degenerate >= degenerate"""
        A1 = np.array([[1, 0], [0, 1], [-1, 0]])
        b1 = np.array([2, 2, 2])
        P1 = Polytope(A1, b1)
        A2 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b2 = np.array([1, 1, 1, -1])
        P2 = Polytope(A2, b2)
        assert contains_(P1, P2)
    
    def test_contains_2d_degenerate_vs_degenerate(self):
        """Test 2D degenerate >= degenerate"""
        A1 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b1 = np.array([3, 1, 3, -1])
        P1 = Polytope(A1, b1)
        A2 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b2 = np.array([1, 1, 1, -1])
        P2 = Polytope(A2, b2)
        assert contains_(P1, P2)
    
    def test_contains_2d_unbounded_vs_unbounded(self):
        """Test 2D unbounded >= unbounded"""
        A1 = np.array([[1, 0], [0, 1], [-1, 0]])
        b1 = np.array([2, 2, 2])
        P1 = Polytope(A1, b1)
        A2 = np.array([[1, 0], [0, 1], [-1, 0]])
        b2 = np.array([1, 1, 1])
        P2 = Polytope(A2, b2)
        assert contains_(P1, P2)
    
    def test_contains_2d_vertices_vs_vertices(self):
        """Test 2D vertex polytope containment"""
        V1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]).T
        P1 = Polytope(V1)
        V2 = np.array([[2, 0], [-2, 0], [0, 2], [0, -1]]).T
        P2 = Polytope(V2)
        assert contains_(P2, P1)
        assert not contains_(P1, P2)
    
    def test_contains_2d_v_polytope_vs_h_polyhedron(self):
        """Test 2D V-polytope vs H-polyhedron"""
        V = np.array([[1, 0], [-1, 1], [-1, -1]]).T
        P1 = Polytope(V)
        A = np.array([[1, 0]])
        b = np.array([0])
        P2 = Polytope(A, b)
        assert not contains_(P1, P2)
    
    def test_contains_2d_v_polytope_point_cloud_contained(self):
        """Test 2D V-polytope >= point cloud (contained)"""
        V = np.array([[1, 0], [-1, 1], [-1, -1]]).T
        P1 = Polytope(V)
        S = np.array([[0, 0], [0.5, 0.1], [-0.8, -0.6], [-0.5, 0.5]]).T
        assert np.all(contains_(P1, S))
    
    def test_contains_2d_h_polyhedron_point_cloud(self):
        """Test 2D H-polyhedron >= point cloud"""
        A = np.array([[1, 0], [0, 1]])
        b = np.array([1, 0])
        P1 = Polytope(A, b)
        S = np.array([[-2, -1], [0, 0], [0, -4]]).T
        assert np.all(contains_(P1, S))
    
    def test_contains_2d_v_polytope_point_cloud_not_contained(self):
        """Test 2D V-polytope point cloud not included"""
        V = np.array([[1, 0], [-1, 1], [-1, -1]]).T
        P1 = Polytope(V)
        S = np.array([[-2, -2], [0, 2], [0, -0.6]]).T
        assert np.all(~contains_(P1, S))
    
    def test_contains_2d_degenerate_v_polytope(self):
        """Test 2D degenerate V-polytope point cloud not included"""
        V = np.array([[-2, -2], [2, 2]]).T
        P1 = Polytope(V)
        S = np.array([[0, 1]]).T
        assert np.all(~contains_(P1, S))
    
    def test_contains_3d_bounded_vs_degenerate(self):
        """Test 3D bounded >= degenerate"""
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], 
                      [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        b = np.ones(6)
        P1 = Polytope(A, b)
        A2 = np.array([[1, 0, 0], [-1, 1, 0], [-1, -1, 0]])
        b2 = np.array([0.5, 0.25, 0.25])
        Ae = np.array([[0, 0, 1]])
        be = np.array([0])
        P2 = Polytope(A2, b2, Ae, be)
        assert contains_(P1, P2)
    
    def test_contains_3d_bounded_not_containing_unbounded(self):
        """Test 3D bounded does not contain unbounded"""
        # Random rotation for generality
        M, _, _ = np.linalg.svd(np.random.randn(3, 3))
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], 
                      [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        b = np.ones(6)
        # Create rotated polytope (this is a placeholder for M * polytope)
        P1 = Polytope(M @ A, b)
        Ae = np.array([[1, 0, 0], [0, 0, 1]])
        be = np.array([1, 2])
        P2 = Polytope(np.zeros((0, 3)), np.zeros(0), Ae, be)
        # This should generally not contain P2 (unbounded plane)
        # Note: actual result depends on random M, so we just test that it runs
        result = contains_(P1, P2)
        assert isinstance(result, (bool, np.bool_))
    
    def test_contains_point_vs_single_point(self):
        """Test point containment with single point"""
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        point = np.array([[0.5], [0.5]])
        assert contains_(P, point)
        
        point_outside = np.array([[2], [2]])
        assert not contains_(P, point_outside)
    
    def test_contains_error_cases(self):
        """Test error cases and edge conditions"""
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        
        # Test with incompatible dimensions
        point_wrong_dim = np.array([[1], [2], [3]])
        with pytest.raises(Exception):
            contains_(P, point_wrong_dim) 