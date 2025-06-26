"""
test_isemptyobject - unit test function for polytope isemptyobject method

Tests the object emptiness checking functionality of polytopes.

Authors: MATLAB: Mark Wetzlinger
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.isemptyobject import isemptyobject


class TestPolytopeIsEmptyObject:
    """Test class for polytope isemptyobject method"""
    
    def test_isemptyobject_2d_no_constraints(self):
        """Test isemptyobject with 2D polytope having no constraints"""
        A = np.zeros((0, 2))
        b = np.zeros(0)
        P = Polytope(A, b)
        assert isemptyobject(P)
    
    def test_isemptyobject_2d_only_inequalities(self):
        """Test isemptyobject with 2D polytope having only inequalities"""
        A = np.array([[-1, 0], [2, 4], [1, -2]])
        b = np.array([-1, 14, -1])
        P = Polytope(A, b)
        assert not isemptyobject(P)
    
    def test_isemptyobject_3d_only_equalities_single_point(self):
        """Test isemptyobject with 3D polytope having only equalities (single point)"""
        Ae = np.array([[1, 0, 1], [0, 1, -1], [1, 0, -1]])
        be = np.array([1, 4, 2])
        P = Polytope(np.zeros((0, 3)), np.zeros(0), Ae, be)
        assert not isemptyobject(P)
    
    def test_isemptyobject_3d_no_vertices(self):
        """Test isemptyobject with 3D polytope having no vertices"""
        V = np.zeros((3, 0))
        P = Polytope(V)
        assert isemptyobject(P)
    
    def test_isemptyobject_1d_no_constraints(self):
        """Test isemptyobject with 1D polytope having no constraints"""
        A = np.zeros((0, 1))
        b = np.zeros(0)
        P = Polytope(A, b)
        assert isemptyobject(P)
    
    def test_isemptyobject_1d_with_constraints(self):
        """Test isemptyobject with 1D polytope having constraints"""
        A = np.array([[1], [-1]])
        b = np.array([2, 1])
        P = Polytope(A, b)
        assert not isemptyobject(P)
    
    def test_isemptyobject_2d_with_vertices(self):
        """Test isemptyobject with 2D polytope defined by vertices"""
        V = np.array([[0, 0], [1, 0], [0, 1]]).T
        P = Polytope(V)
        assert not isemptyobject(P)
    
    def test_isemptyobject_2d_empty_vertices(self):
        """Test isemptyobject with 2D polytope having empty vertices"""
        V = np.zeros((2, 0))
        P = Polytope(V)
        assert isemptyobject(P)
    
    def test_isemptyobject_mixed_constraints(self):
        """Test isemptyobject with mixed inequality and equality constraints"""
        # Non-empty case
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        Ae = np.array([[0, 1]])
        be = np.array([0])
        P = Polytope(A, b, Ae, be)
        assert not isemptyobject(P)
    
    def test_isemptyobject_high_dimensional(self):
        """Test isemptyobject with higher dimensional polytopes"""
        # 4D polytope with no constraints
        A = np.zeros((0, 4))
        b = np.zeros(0)
        P = Polytope(A, b)
        assert isemptyobject(P)
        
        # 4D polytope with constraints
        A = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        assert not isemptyobject(P)
    
    def test_isemptyobject_edge_cases(self):
        """Test edge cases for isemptyobject"""
        # Single vertex
        V = np.array([[1, 2, 3]]).T
        P = Polytope(V)
        assert not isemptyobject(P)
        
        # Line segment
        V = np.array([[0, 0], [1, 1]]).T
        P = Polytope(V)
        assert not isemptyobject(P)
        
        # Empty matrix with correct dimensions
        V = np.zeros((5, 0))
        P = Polytope(V)
        assert isemptyobject(P)
    
    def test_isemptyobject_consistency(self):
        """Test consistency of isemptyobject across different representations"""
        # Same polytope represented differently should give same result
        
        # H-representation of unit square
        A_h = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b_h = np.array([1, 1, 1, 1])
        P_h = Polytope(A_h, b_h)
        
        # V-representation of unit square
        V_v = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]).T
        P_v = Polytope(V_v)
        
        # Both should not be empty objects
        assert not isemptyobject(P_h)
        assert not isemptyobject(P_v)
        
        # Empty representations should both be empty
        A_empty = np.zeros((0, 2))
        b_empty = np.zeros(0)
        P_h_empty = Polytope(A_empty, b_empty)
        
        V_empty = np.zeros((2, 0))
        P_v_empty = Polytope(V_empty)
        
        assert isemptyobject(P_h_empty)
        assert isemptyobject(P_v_empty)
    
    def test_isemptyobject_different_dimensions(self):
        """Test isemptyobject across different dimensions"""
        for dim in range(1, 6):
            # Empty polytope
            A_empty = np.zeros((0, dim))
            b_empty = np.zeros(0)
            P_empty = Polytope(A_empty, b_empty)
            assert isemptyobject(P_empty)
            
            # Non-empty polytope (hypercube)
            if dim > 0:
                A_cube = np.vstack([np.eye(dim), -np.eye(dim)])
                b_cube = np.ones(2*dim)
                P_cube = Polytope(A_cube, b_cube)
                assert not isemptyobject(P_cube)
    
    def test_isemptyobject_after_operations(self):
        """Test isemptyobject after various operations"""
        # Create non-empty polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        assert not isemptyobject(P)
        
        # After copying, should still not be empty
        P_copy = Polytope(P.A.copy(), P.b.copy())
        assert not isemptyobject(P_copy)
    
    def test_isemptyobject_degenerate_cases(self):
        """Test isemptyobject with degenerate cases"""
        # Point polytope
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([1, 2])
        P_point = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert not isemptyobject(P_point)
        
        # Line polytope
        Ae = np.array([[1, 1]])
        be = np.array([1])
        P_line = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert not isemptyobject(P_line) 