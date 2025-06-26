"""
test_dim - unit test function for polytope dim method

Tests the dimension determination functionality of polytopes.

Authors: MATLAB: Viktor Kotsev
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.dim import dim


class TestPolytopeDim:
    """Test class for polytope dim method"""
    
    def test_dim_empty_polytope(self):
        """Test empty polytope dimension"""
        # Using the empty constructor if available, otherwise construct empty polytope
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([-1, -1])  # Contradictory: x >= 1 and x <= -1
        P_empty = Polytope(A, b)  # Empty polytope in 2D
        assert dim(P_empty) == 2
    
    def test_dim_1d_bounded_degenerate(self):
        """Test 1D, bounded, degenerate (single point)"""
        A = np.array([[1], [-1]])
        b = np.array([1, -1])
        P = Polytope(A, b)
        assert dim(P) == 1
    
    def test_dim_1d_empty(self):
        """Test 1D, empty"""
        A = np.array([[1], [-1]])
        b = np.array([2, -3])
        P = Polytope(A, b)
        assert dim(P) == 1
    
    def test_dim_1d_unbounded(self):
        """Test 1D, unbounded"""
        A = np.array([[1]])
        b = np.array([1])
        P = Polytope(A, b)
        assert dim(P) == 1
    
    def test_dim_1d_single_point_equality_constraints(self):
        """Test 1D, single point, only equality constraints"""
        Ae = np.array([[1]])
        be = np.array([3])
        P = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)
        assert dim(P) == 1
    
    def test_dim_1d_vertex_instantiation(self):
        """Test 1D, vertex instantiation"""
        V = np.array([[-1, 2, 4]])
        P = Polytope(V)
        assert dim(P) == 1
    
    def test_dim_2d_bounded_non_degenerate(self):
        """Test 2D, bounded, non-degenerate"""
        A = np.array([[-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([2, 3, 2, 3, 2])
        P = Polytope(A, b)
        assert dim(P) == 2
    
    def test_dim_2d_bounded_degenerate(self):
        """Test 2D, bounded, degenerate"""
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        Ae = np.array([[0, 1]])
        be = np.array([1])
        P = Polytope(A, b, Ae, be)
        assert dim(P) == 2
    
    def test_dim_2d_unbounded_non_degenerate(self):
        """Test 2D, unbounded, non-degenerate"""
        A = np.array([[1, 0], [0, 1], [-1, 0]])
        b = np.array([2, 2, 2])
        P = Polytope(A, b)
        assert dim(P) == 2
    
    def test_dim_2d_unbounded_degenerate(self):
        """Test 2D, unbounded, degenerate"""
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, -1])
        P = Polytope(A, b)
        assert dim(P) == 2
    
    def test_dim_2d_empty_equality_constraints(self):
        """Test 2D, empty, only equality constraints"""
        Ae = np.array([[0, 1], [1, 0], [1, 1]])
        be = np.array([1, 1, 1])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert dim(P) == 2
    
    def test_dim_2d_vertex_instantiation(self):
        """Test 2D, vertex instantiation"""
        V = np.array([[1, 0], [1, 1], [-2, 1], [-2, -2], [0, -2]]).T
        P = Polytope(V)
        assert dim(P) == 2
    
    def test_dim_3d_fully_empty_inequalities(self):
        """Test 3D, fully empty (no constraints)"""
        A = np.zeros((0, 3))
        b = np.zeros(0)
        P = Polytope(A, b)
        assert dim(P) == 3
    
    def test_dim_3d_fully_empty_equalities(self):
        """Test 3D, fully empty with equality constraints"""
        Ae = np.zeros((0, 3))
        be = np.zeros(0)
        P = Polytope(np.zeros((0, 3)), np.zeros(0), Ae, be)
        assert dim(P) == 3
    
    def test_dim_4d_box(self):
        """Test 4D, box (hypercube)"""
        A = np.vstack([np.eye(4), -np.eye(4)])
        b = np.ones(8)
        P = Polytope(A, b)
        assert dim(P) == 4
    
    def test_dim_high_dimensional(self):
        """Test higher dimensional polytopes"""
        # 5D hypercube
        n = 5
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.ones(2*n)
        P = Polytope(A, b)
        assert dim(P) == 5
        
        # 10D with only a few constraints
        n = 10
        A = np.array([[1] + [0]*9])  # Only one constraint: x1 <= 1
        b = np.array([1])
        P = Polytope(A, b)
        assert dim(P) == 10
    
    def test_dim_mixed_constraints(self):
        """Test dimension with mixed inequality and equality constraints"""
        # 3D polytope with one equality constraint (making it 2D surface in 3D space)
        A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        b = np.array([1, 1, 1, 1])
        Ae = np.array([[0, 0, 1]])
        be = np.array([0])
        P = Polytope(A, b, Ae, be)
        # The ambient dimension is still 3, even though the polytope is 2D
        assert dim(P) == 3
    
    def test_dim_single_point_various_dimensions(self):
        """Test dimension of single points in various dimensions"""
        # 2D single point
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([1, 2])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert dim(P) == 2
        
        # 3D single point
        Ae = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        be = np.array([1, 2, 3])
        P = Polytope(np.zeros((0, 3)), np.zeros(0), Ae, be)
        assert dim(P) == 3
    
    def test_dim_degenerate_cases(self):
        """Test dimension of various degenerate cases"""
        # Line in 2D (one equality constraint)
        Ae = np.array([[1, 1]])
        be = np.array([1])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert dim(P) == 2  # Ambient dimension
        
        # Plane in 3D (one equality constraint)
        Ae = np.array([[1, 1, 1]])
        be = np.array([1])
        P = Polytope(np.zeros((0, 3)), np.zeros(0), Ae, be)
        assert dim(P) == 3  # Ambient dimension
    
    def test_dim_edge_cases(self):
        """Test edge cases for dimension"""
        # Very small 1D polytope
        A = np.array([[1], [-1]])
        b = np.array([1e-10, 1e-10])
        P = Polytope(A, b)
        assert dim(P) == 1
        
        # Large dimensional space with simple constraint
        n = 20
        A = np.array([[1] + [0]*(n-1)])
        b = np.array([1])
        P = Polytope(A, b)
        assert dim(P) == n
    
    def test_dim_vertex_polytopes_various_dimensions(self):
        """Test dimension of vertex polytopes in various dimensions"""
        # 1D line segment
        V = np.array([[-1, 1]])
        P = Polytope(V)
        assert dim(P) == 1
        
        # 2D triangle
        V = np.array([[0, 1, 0], [0, 0, 1]])
        P = Polytope(V)
        assert dim(P) == 2
        
        # 3D tetrahedron
        V = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        P = Polytope(V)
        assert dim(P) == 3
    
    def test_dim_consistency(self):
        """Test that dimension is consistent across different representations"""
        # Create same polytope using H-representation and V-representation
        # Unit square
        A_h = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b_h = np.array([1, 1, 1, 1])
        P_h = Polytope(A_h, b_h)
        
        V_v = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]).T
        P_v = Polytope(V_v)
        
        assert dim(P_h) == dim(P_v) == 2 