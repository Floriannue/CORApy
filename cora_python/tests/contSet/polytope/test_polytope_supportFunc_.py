"""
Test file for polytope supportFunc_ method

This file contains unit tests for the polytope supportFunc_ method.
Mirrors MATLAB test_polytope_supportFunc.m
"""

import pytest
import numpy as np
from cora_python.contSet.polytope import Polytope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestPolytopeSupportFunc:
    """Test class for polytope supportFunc_ method"""
    
    def test_supportFunc_1d_bounded(self):
        """Test supportFunc_ for 1D bounded polytope"""
        # 1D, bounded
        A = np.array([[1], [-1]])
        b = np.array([2, 0.5])
        P = Polytope(A, b)
        
        # Test upper bound
        val = P.supportFunc_(np.array([1]), 'upper')
        assert val == 2
        
        # Test lower bound
        val = P.supportFunc_(np.array([1]), 'lower')
        assert val == -0.5
    
    def test_supportFunc_1d_unbounded(self):
        """Test supportFunc_ for 1D unbounded polytope"""
        # 1D, unbounded
        A = np.array([[1]])
        b = np.array([1])
        P = Polytope(A, b)
        
        # Test upper bound
        val = P.supportFunc_(np.array([1]), 'upper')
        assert val == 1
        
        # Test negative direction (should be unbounded)
        val = P.supportFunc_(np.array([-1]), 'upper')
        assert np.isinf(val)
    
    def test_supportFunc_1d_empty(self):
        """Test supportFunc_ for 1D empty polytope"""
        # 1D, fully empty (no constraints)
        A = np.zeros((0, 1))
        b = np.zeros((0, 0))
        P = Polytope(A, b)
        
        # Test both directions (should be unbounded)
        val = P.supportFunc_(np.array([1]), 'upper')
        assert np.isinf(val)
        
        val = P.supportFunc_(np.array([-1]), 'upper')
        assert np.isinf(val)
    
    def test_supportFunc_2d_infeasible(self):
        """Test supportFunc_ for 2D polytope with infeasible constraints"""
        # 2D, infeasible constraints -> empty
        A = np.array([[-1, -1], [-1, 1], [1, 0]])
        b = np.array([-2, -2, -1])
        P = Polytope(A, b)
        
        # Test support function in x-direction
        val = P.supportFunc_(np.array([1, 0]), 'upper')
        assert np.isinf(val) and val < 0  # Should be -inf
        
        val = P.supportFunc_(np.array([1, 0]), 'lower')
        assert np.isinf(val) and val > 0  # Should be +inf
    
    def test_supportFunc_2d_vertex_representation(self):
        """Test supportFunc_ for 2D polytope in vertex representation"""
        # 2D, vertex representation
        V = np.array([[1, 0.8, 0, -0.5, 0.2],
                      [0, 1, 1.4, 0.5, -1]])
        P = Polytope(V)
        
        # Test upper bound in x-direction
        val, x = P.supportFunc_(np.array([1, 0]), 'upper')
        assert withinTol(val, 1)
        assert np.allclose(x, np.array([1, 0]))
        
        # Test lower bound in x-direction
        val, x = P.supportFunc_(np.array([1, 0]), 'lower')
        assert withinTol(val, -0.5)
        assert np.allclose(x, np.array([-0.5, 0.5]))
        
        # Test range in x-direction
        val, x = P.supportFunc_(np.array([1, 0]), 'range')
        assert val.inf == -0.5 and val.sup == 1
        # Note: x output format may differ from MATLAB
    
    def test_supportFunc_2d_normalized(self):
        """Test supportFunc_ for 2D polytope with normalized constraints"""
        # 2D, no redundant halfspaces
        A = np.array([[2, 1], [1, 3], [2, -2], [-2, -2], [-1, 2]])
        # Normalize A by dividing each row by its norm
        norms = np.linalg.norm(A, axis=1)
        A = A / norms[:, np.newaxis]
        b = np.array([2, 1, 2, 2, 2])
        P = Polytope(A, b)
        
        # Compute support function along chosen halfspaces
        for i in range(len(b)):
            sF = P.supportFunc_(A[i, :], 'upper')
            assert withinTol(sF, b[i])
    
    def test_supportFunc_2d_unbounded(self):
        """Test supportFunc_ for 2D unbounded polytope"""
        # 2D, unbounded
        A = np.array([[1, 0], [-1, 0]])
        b = np.ones((2, 1))
        P = Polytope(A, b)
        
        # Test y-direction (should be unbounded)
        val = P.supportFunc_(np.array([0, 1]), 'upper')
        assert np.isinf(val)
        
        val = P.supportFunc_(np.array([0, -1]), 'upper')
        assert np.isinf(val)
        
        val = P.supportFunc_(np.array([0, 1]), 'lower')
        assert np.isinf(val) and val < 0  # Should be -inf
        
        val = P.supportFunc_(np.array([0, -1]), 'lower')
        assert np.isinf(val) and val < 0  # Should be -inf
    
    def test_supportFunc_2d_zonotope_conversion(self):
        """Test supportFunc_ for 2D polytope converted from zonotope"""
        # 2D, comparison to support function values from other representation
        # Create a zonotope and convert to polytope
        from cora_python.contSet.zonotope import Zonotope
        c = np.array([1, 1])
        G = np.array([[1.71, -2.14, 1.35, 0.96],
                      [-0.19, -0.84, -1.07, 0.12]])
        Z = Zonotope(c, G)
        P = Polytope(Z)
        
        tol = 1e-6
        
        # Test various directions
        assert withinTol(P.supportFunc_(np.array([0, 1]), 'upper'), 3.22, tol)
        assert withinTol(P.supportFunc_(np.array([1, 0]), 'upper'), 7.16, tol)
        assert withinTol(P.supportFunc_(np.array([0, -1]), 'upper'), 1.22, tol)
        assert withinTol(P.supportFunc_(np.array([-1, 0]), 'upper'), 5.16, tol)
        assert withinTol(P.supportFunc_(np.array([1, 1]), 'upper'), 7.86, tol)
        assert withinTol(P.supportFunc_(np.array([-1, -1]), 'upper'), 3.86, tol)
        assert withinTol(P.supportFunc_(np.array([-1, 1]), 'upper'), 6.46, tol)
        assert withinTol(P.supportFunc_(np.array([1, -1]), 'upper'), 6.46, tol)
    
    def test_supportFunc_2d_redundant_halfspaces(self):
        """Test supportFunc_ for 2D polytope with and without redundant halfspaces"""
        # 2D, without/with redundant halfspaces
        A = np.array([[2, 1], [-2, 1], [-1, -2]])
        b = np.array([2, 1, 2])
        P = Polytope(A, b)
        
        # Add redundant halfspaces
        A_ = np.vstack([A, np.array([[-2, -3], [1, 1.5], [-2, 1.5]])])
        b_ = np.vstack([b.reshape(-1, 1), np.array([[4], [3], [3]])])
        P_ = Polytope(A_, b_)
        
        # Both should give the same support function values
        direction = np.array([1, 0])
        val1 = P.supportFunc_(direction, 'upper')
        val2 = P_.supportFunc_(direction, 'upper')
        assert withinTol(val1, val2)
    
    def test_supportFunc_edge_cases(self):
        """Test supportFunc_ for edge cases"""
        # Test with zero direction
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        # Zero direction should return 0
        val = P.supportFunc_(np.array([0, 0]), 'upper')
        assert val == 0
        
        # Test with very small direction
        val = P.supportFunc_(np.array([1e-10, 1e-10]), 'upper')
        assert val >= 0  # Should be non-negative for small directions
