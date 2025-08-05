"""
test_center - unit test function for polytope center method

Tests the center computation functionality of polytopes,
including different methods for center computation.

Authors: MATLAB: Viktor Kotsev, Mark Wetzlinger, Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope

class TestPolytopeCenter:
    """Test class for polytope center method"""
    
    def test_center_1d_bounded_inequalities_only(self):
        """Test 1D, only inequalities, bounded"""
        A = np.array([[2], [-1]])
        b = np.array([6, 1])
        P = Polytope(A, b)
        c = P.center()
        c_true = np.array([1])
        assert np.allclose(c, c_true)
    
    def test_center_1d_single_point_equalities_only(self):
        """Test 1D, only equalities, single point"""
        Ae = np.array([[3]])
        be = np.array([5])
        P = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)
        c = P.center()
        c_true = np.array([5/3])
        assert np.allclose(c, c_true)
    
    def test_center_1d_unbounded_inequalities_only(self):
        """Test 1D, only inequalities, unbounded"""
        A = np.array([[3], [2], [4]])
        b = np.array([5, 2, -3])
        P = Polytope(A, b)
        c = P.center()
        assert np.all(np.isnan(c))
    
    def test_center_1d_empty_equalities(self):
        """Test 1D, only inequalities, empty"""
        Ae = np.array([[1], [4]])
        be = np.array([2, -5])
        P = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)
        c = P.center()
        assert c.size == 0  # Empty center
    
    def test_center_1d_empty_mixed_constraints(self):
        """Test 1D, inequalities and equalities, empty"""
        A = np.array([[1], [-4]])
        b = np.array([4, -2])
        Ae = np.array([[5]])
        be = np.array([100])
        P = Polytope(A, b, Ae, be)
        c = P.center()
        assert c.size == 0  # Empty center
    
    def test_center_1d_fully_empty(self):
        """Test 1D, fully empty"""
        A = np.zeros((0, 1))
        b = np.zeros(0)
        P = Polytope(A, b)
        c = P.center()
        assert c.size == 0
    
    def test_center_2d_bounded_inequalities_only(self):
        """Test 2D, only inequalities, bounded"""
        A = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        b = np.ones(4)
        P = Polytope(A, b)
        c = P.center()
        c_true = np.array([0, 0])
        assert np.allclose(c, c_true)
    
    def test_center_2d_empty_inequalities(self):
        """Test 2D, only inequalities, empty"""
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([-1, -1])
        P = Polytope(A, b)
        c = P.center()
        assert c.size == 0  # Empty center
    
    def test_center_2d_empty_equalities_only(self):
        """Test 2D, only equalities, empty"""
        Ae = np.array([[1, 0], [0, 1], [0, 1]])
        be = np.array([1, -1, 0])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        c = P.center()
        assert c.size == 0  # Empty center
    
    def test_center_2d_single_point_equalities_only(self):
        """Test 2D, only equalities, single point"""
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([0, 0])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        c = P.center()
        c_true = np.array([0, 0])
        assert np.allclose(c, c_true)
    
    def test_center_2d_unbounded_mixed_constraints(self):
        """Test 2D, inequalities and equalities, unbounded"""
        A = np.array([[1, 0]])
        b = np.array([1])
        Ae = np.array([[0, 1]])
        be = np.array([1])
        P = Polytope(A, b, Ae, be)
        c = P.center()
        assert np.all(np.isnan(c))
    
    def test_center_2d_bounded_mixed_constraints(self):
        """Test 2D, inequalities and equalities, bounded"""
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        Ae = np.array([[0, 1]])
        be = np.array([1])
        P = Polytope(A, b, Ae, be)
        c = P.center()
        c_true = np.array([0, 1])
        assert np.allclose(c, c_true)
    
    def test_center_2d_empty_mixed_constraints(self):
        """Test 2D, inequalities and equalities, empty"""
        A = np.array([[2, 1], [-1, 2], [0, -1]])
        b = np.ones(3)
        Ae = np.array([[1, 1]])
        be = np.array([10])
        P = Polytope(A, b, Ae, be)
        c = P.center()
        assert c.size == 0  # Empty center
    
    def test_center_2d_fully_empty_equalities(self):
        """Test 2D, fully empty"""
        Ae = np.zeros((0, 2))
        be = np.zeros(0)
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        c = P.center()
        assert c.size == 0
    
    def test_center_2d_v_polytope(self):
        """Test 2D, V-polytope"""
        V = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]).T
        P = Polytope(V)
        c = P.center()
        c_true = np.array([0, 0])
        assert np.allclose(c, c_true)
    
    def test_center_3d_bounded_inequalities_only(self):
        """Test 3D, only inequalities, bounded"""
        A = np.array([[0, 1, 0], [0, 0, 1], [0, -1, 0], 
                      [0, 0, -1], [1, 0, 0], [-1, 0, 0]])
        b = np.ones(6)
        P = Polytope(A, b)
        c = P.center()
        c_true = np.array([0, 0, 0])
        assert np.allclose(c, c_true)
    
    def test_center_3d_bounded_degenerate_mixed_constraints(self):
        """Test 3D, inequalities and equalities, bounded, degenerate"""
        A = np.array([[0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]])
        b = np.ones(4)
        Ae = np.array([[1, 0, 0]])
        be = np.array([2])
        P = Polytope(A, b, Ae, be)
        c = P.center()
        c_true = np.array([2, 0, 0])
        assert np.allclose(c, c_true)
    
    def test_center_3d_unbounded_inequalities_only(self):
        """Test 3D, only inequalities, unbounded"""
        A = np.array([[1, 0, 0], [0, 1, 0]])
        b = np.array([0, 0])
        P = Polytope(A, b)
        c = P.center()
        assert np.all(np.isnan(c))
    
    def test_center_3d_unbounded_equalities_only(self):
        """Test 3D, only equalities, unbounded"""
        Ae = np.array([[1, 0, 0], [0, 1, 0]])
        be = np.array([0, 0])
        P = Polytope(np.zeros((0, 3)), np.zeros(0), Ae, be)
        c = P.center()
        assert np.all(np.isnan(c))
    
    def test_center_3d_unbounded_mixed_constraints(self):
        """Test 3D, inequalities and equalities, unbounded"""
        A = np.array([[1, 1, 0], [-1, 0, 0]])
        b = np.array([1, 1])
        Ae = np.array([[0, 1, 1]])
        be = np.array([1])
        P = Polytope(A, b, Ae, be)
        c = P.center()
        assert np.all(np.isnan(c))
    
    def test_center_3d_single_point(self):
        """Test 3D, single point"""
        Ae = np.array([[1, 2, -1], [0, 1, 1], [-1, 2, 1]])
        be = np.array([1, 1, 1])
        P = Polytope(np.zeros((0, 3)), np.zeros(0), Ae, be)
        c = P.center()
        c_true = np.array([0.5, 0.5, 0.5])
        assert np.allclose(c, c_true)
    
    def test_center_v_polytope_line(self):
        """Test V-polytope, line"""
        V = np.array([[1, 3], [2, 4]])
        P = Polytope(V)
        c = P.center(method='avg')
        c_true = np.array([2, 3])
        assert np.allclose(c, c_true)
    
    def test_center_v_polytope_1d(self):
        """Test V-polytope, 1D"""
        V = np.array([[1, 3]])
        P = Polytope(V)
        c = P.center(method='avg')
        c_true = np.array([2])
        assert np.array_equal(c, c_true)
    
    def test_center_v_polytope_2d_avg(self):
        """Test V-polytope, 2D, average method"""
        V = np.array([[1, 0, 1], [0, 1, 1]])
        P = Polytope(V)
        c = P.center(method='avg')
        c_true = np.array([2/3, 2/3])
        assert np.allclose(c, c_true)
    
    def test_center_v_polytope_2d_chebyshev(self):
        """Test V-polytope, 2D, Chebyshev method"""
        V = np.array([[1, 0, 1], [0, 1, 1]])
        P = Polytope(V)
        c = P.center(method='chebyshev')
        c_true = np.array([0.7071067811865472, 0.7071067811865475])
        assert np.allclose(c, c_true)
    
    def test_center_v_polytope_3d_avg(self):
        """Test V-polytope, 3D, average method"""
        V = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
        P = Polytope(V)
        c = P.center(method='avg')
        c_true = np.array([0.75, 0.75, 0.25])
        assert np.allclose(c, c_true)
    
    def test_center_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test with very small polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        P = Polytope(A, b)
        c = P.center()
        # Should be close to origin for very small polytope
        assert np.allclose(c, np.array([0, 0]), atol=1e-9)
        
    def test_center_method_parameter(self):
        """Test different method parameters"""
        V = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]).T
        P = Polytope(V)
        
        # Test default method
        c_default = P.center()
        
        # Test explicit chebyshev method
        c_chebyshev = P.center(method='chebyshev')
        
        # Test average method for V-polytope
        c_avg = P.center(method='avg')
        
        # All should give reasonable results for this symmetric polytope
        assert np.allclose(c_default, np.array([0, 0]), atol=1e-6)
        assert np.allclose(c_chebyshev, np.array([0, 0]), atol=1e-6)
        assert np.allclose(c_avg, np.array([0, 0]), atol=1e-6) 