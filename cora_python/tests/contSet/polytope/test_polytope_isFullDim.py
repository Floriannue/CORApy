"""
test_polytope_isFullDim - unit test function of isFullDim

Tests the full-dimensional determination functionality of polytopes.

Authors: Viktor Kotsev, Adrian Kulmburg, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 09-May-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.isFullDim import isFullDim


class TestPolytopeIsFullDim:
    """Test class for polytope isFullDim method"""
    
    def test_isFullDim_empty_object(self):
        """Test empty object"""
        P = Polytope.empty(2)
        assert not isFullDim(P) and P._fullDim_val is not None and P._fullDim_val == False
    
    def test_isFullDim_fullspace_object(self):
        """Test fullspace object"""
        P = Polytope.Inf(2)
        assert isFullDim(P) and P._fullDim_val is not None and P._fullDim_val == True
    
    def test_isFullDim_1d_bounded(self):
        """Test 1D, only inequalities, bounded"""
        A = np.array([[2], [-1]])
        b = np.array([6, 1])
        P = Polytope(A, b)
        
        # Verify cache is not set initially
        assert P._fullDim_val is None
        
        # Call isFullDim and verify result
        result = isFullDim(P)
        assert result == True
        
        # Verify cache is now set correctly like MATLAB (P.fullDim.val = res)
        assert P._fullDim_val is not None and P._fullDim_val == True
    
    def test_isFullDim_1d_single_point(self):
        """Test 1D, only equalities, single point"""
        Ae = np.array([[3]])
        be = np.array([5])
        P = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)
        
        # Verify cache is not set initially
        assert P._fullDim_val is None
        
        # Call isFullDim and verify result
        result = isFullDim(P)
        assert result == False
        
        # Verify cache is now set correctly
        assert P._fullDim_val is not None and P._fullDim_val == False
    
    def test_isFullDim_1d_unbounded(self):
        """Test 1D, only inequalities, unbounded"""
        A = np.array([[3], [2], [4]])
        b = np.array([5, 2, -3])
        P = Polytope(A, b)
        
        result = isFullDim(P)
        assert result == True and P._fullDim_val is not None and P._fullDim_val == True
    
    def test_isFullDim_1d_empty_equalities(self):
        """Test 1D, only inequalities, empty"""
        Ae = np.array([[1], [4]])
        be = np.array([2, -5])
        P = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)
        
        result = isFullDim(P)
        assert (result == False and P._fullDim_val is not None and P._fullDim_val == False and
                P._emptySet_val is not None and P._emptySet_val == True)
    
    def test_isFullDim_1d_mixed_empty(self):
        """Test 1D, inequalities and equalities, empty"""
        A = np.array([[1], [-4]])
        b = np.array([4, -2])
        Ae = np.array([[5]])
        be = np.array([100])
        P = Polytope(A, b, Ae, be)
        
        result = isFullDim(P)
        assert (result == False and P._fullDim_val is not None and P._fullDim_val == False and
                P._emptySet_val is not None and P._emptySet_val == True)
    
    def test_isFullDim_1d_fully_empty(self):
        """Test 1D, fully empty"""
        A = np.zeros((0, 1))
        b = np.zeros(0)
        P = Polytope(A, b)
        
        result = isFullDim(P)
        assert (result == True and P._fullDim_val is not None and P._fullDim_val == True and
                P._emptySet_val is not None and P._emptySet_val == False)
    
    def test_isFullDim_1d_vertex_instantiation(self):
        """Test 1D, vertex instantiation"""
        V = np.array([[1, 2]]).T
        P = Polytope(V)
        
        # MATLAB: [res_,X] = isFullDim(P); assert(~res_ && isempty(X));
        result, X = isFullDim(P, return_subspace=True)
        assert result == False and X.size == 0
    
    def test_isFullDim_2d_empty(self):
        """Test 2D, empty"""
        A = np.array([[1, 0]])
        b = np.array([3])
        Ae = np.array([[1, 0]])
        be = np.array([4])
        P = Polytope(A, b, Ae, be)
        
        result = isFullDim(P)
        assert result == False and P._fullDim_val is not None and P._fullDim_val == False
    
    def test_isFullDim_2d_nondegenerate_vertex(self):
        """Test 2D, non-degenerate, vertex instantiation"""
        V = np.array([[2, 0], [-2, 0], [0, 2], [0, -2]]).T
        P = Polytope(V)
        
        result = isFullDim(P)
        assert result == True and P._fullDim_val is not None and P._fullDim_val == True
    
    def test_isFullDim_2d_nondegenerate_bounded(self):
        """Test 2D, non-degenerate, bounded"""
        A = np.array([[-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([2, 3, 2, 3, 2])
        P = Polytope(A, b)
        
        result = isFullDim(P)
        assert result == True and P._fullDim_val is not None and P._fullDim_val == True
    
    def test_isFullDim_2d_degenerate_case1(self):
        """Test 2D, degenerate case 1"""
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([2, 2, 2, -2])
        P = Polytope(A, b)
        
        result = isFullDim(P)
        assert result == False and P._fullDim_val is not None and P._fullDim_val == False
    
    def test_isFullDim_2d_degenerate_case2(self):
        """Test 2D, degenerate case 2"""
        A = np.array([[1, 1], [1, -1], [-1, 0]])
        b = np.zeros(3)
        P = Polytope(A, b)
        
        result = isFullDim(P)
        assert result == False and P._fullDim_val is not None and P._fullDim_val == False
    
    def test_isFullDim_cache_reuse(self):
        """Test that cached values are reused"""
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        
        # First call
        result1 = isFullDim(P)
        cache_val = P._fullDim_val
        
        # Second call should use cached value
        result2 = isFullDim(P)
        
        assert result1 == result2 == True
        assert P._fullDim_val == cache_val == True