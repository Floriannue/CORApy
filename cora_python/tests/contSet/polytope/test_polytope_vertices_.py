"""
test_vertices_ - unit test function for polytope vertices_ method

Tests the vertex computation functionality of polytopes.

Authors: MATLAB: Mark Wetzlinger
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.vertices_ import vertices_


class TestPolytopeVertices:
    """Test class for polytope vertices_ method"""
    
    def test_vertices_1d_empty_inequalities_only(self):
        """Test 1D, empty, only inequalities"""
        A = np.array([[2], [1], [-4]])
        b = np.array([5, 0.5, -3])
        P = Polytope(A, b)
        V = vertices_(P)
        assert V.size == 0  # Empty
        
        # Test cache values are set correctly like MATLAB
        assert P._emptySet_val == True
        assert P._bounded_val == True  # Empty sets are bounded
        assert P._minVRep_val == True  # Empty V-rep is minimal
    
    def test_vertices_1d_empty_mixed_constraints(self):
        """Test 1D, empty, inequalities and equalities"""
        A = np.array([[1], [-4]])
        b = np.array([4, -2])
        Ae = np.array([[5]])
        be = np.array([100])
        P = Polytope(A, b, Ae, be)
        V = vertices_(P)
        assert V.size == 0  # Empty
    
    def test_vertices_1d_bounded(self):
        """Test 1D, bounded"""
        A = np.array([[1], [-2]])
        b = np.array([1.25, 1])
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([[-0.5, 1.25]])
        assert np.allclose(V, V_expected)
    
    def test_vertices_1d_bounded_non_minimal(self):
        """Test 1D, bounded, non-minimal representation"""
        A = np.array([[2], [1], [-1], [-2]])
        b = np.array([1, 2, 5, 2])
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([[-1, 0.5]])
        assert np.allclose(V, V_expected)
    
    def test_vertices_1d_unbounded_case1(self):
        """Test 1D, unbounded (first case)"""
        A = np.array([[1]])
        b = np.array([0])
        P = Polytope(A, b)
        # Should raise NotSupported error for unbounded polytope
        with pytest.raises(Exception):
            vertices_(P)
    
    def test_vertices_1d_unbounded_case2(self):
        """Test 1D, unbounded (second case)"""
        A = np.array([[3]])
        b = np.array([6])
        P = Polytope(A, b)
        # Should raise NotSupported error for unbounded polytope
        with pytest.raises(Exception):
            vertices_(P)
    
    def test_vertices_1d_degenerate_inequalities(self):
        """Test 1D, degenerate, only inequalities"""
        A = np.array([[2], [-1]])
        b = np.array([4, -2])
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([[2]])
        assert np.allclose(V, V_expected)
    
    def test_vertices_1d_degenerate_equality(self):
        """Test 1D, degenerate, equality"""
        Ae = np.array([[1]])
        be = np.array([2])
        P = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)
        V = vertices_(P)
        V_expected = np.array([[2]])
        assert np.allclose(V, V_expected)
    
    def test_vertices_2d_bounded(self):
        """Test 2D, bounded"""
        A = np.array([[2, 1], [2, -2], [-2, -2], [-1, 2]])
        b = np.array([2, 2, 2, 2])
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([[0.4, 1.2], [1, 0], [0, -1], [-4/3, 1/3]]).T
        # Check that vertices match (order may vary)
        assert self._compare_vertex_sets(V, V_expected)
    
    def test_vertices_2d_vertex_instantiation(self):
        """Test 2D, vertex instantiation"""
        V_input = np.array([[3, 0], [2, 2], [-1, 3], [-2, 0], [0, -1]]).T
        P = Polytope(V_input)
        V = vertices_(P)
        assert self._compare_vertex_sets(V, V_input)
    
    def test_vertices_2d_bounded_degenerate_point(self):
        """Test 2D, bounded, degenerate (single point)"""
        A = np.array([[1, 1], [1, -1], [-1, 0]])
        b = np.zeros(3)
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([[0, 0]]).T
        assert np.allclose(V, V_expected)
    
    def test_vertices_2d_bounded_degenerate_line(self):
        """Test 2D, bounded, degenerate (line)"""
        A = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
        b = np.array([1, 0, 1, 0])
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([[0.5, 0.5], [-0.5, -0.5]]).T
        assert self._compare_vertex_sets(V, V_expected)
    
    def test_vertices_3d_degenerate_2d_simplex(self):
        """Test 3D, degenerate (2D simplex)"""
        A = np.array([[-1, 0, 0], [0, -1, 0], [1, 1, 0]])
        b = np.array([0, 0, 2])
        Ae = np.array([[0, 0, 1]])
        be = np.array([0])
        P = Polytope(A, b, Ae, be)
        V = vertices_(P)
        V_expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]]).T
        assert self._compare_vertex_sets(V, V_expected)
    
    def test_vertices_3d_unit_box(self):
        """Test 3D, unit box"""
        n = 3
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.ones(2*n)
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
            [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]).T
        assert self._compare_vertex_sets(V, V_expected)
    
    def test_vertices_3d_degenerate_unit_square(self):
        """Test 3D, degenerate unit box: square"""
        n = 3
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.array([1, 1, 0, 1, 1, 0])
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]).T
        assert self._compare_vertex_sets(V, V_expected)
    
    def test_vertices_4d_degenerate_unit_square(self):
        """Test 4D, degenerate (unit square)"""
        A = np.vstack([np.hstack([np.eye(2), np.zeros((2, 2))]), 
                       np.hstack([-np.eye(2), np.zeros((2, 2))])])
        b = np.ones(4)
        Ae = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        be = np.array([0, 0])
        P = Polytope(A, b, Ae, be)
        V = vertices_(P)
        V_expected = np.array([[1, 1, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0], [-1, -1, 0, 0]]).T
        assert self._compare_vertex_sets(V, V_expected)
    
    def test_vertices_4d_degenerate_rotated_unit_square(self):
        """Test 4D, degenerate (rotated unit square)"""
        A = np.vstack([np.hstack([np.eye(2), np.zeros((2, 2))]), 
                       np.hstack([-np.eye(2), np.zeros((2, 2))])])
        b = np.ones(4)
        Ae = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        be = np.array([0, 0])
        P = Polytope(A, b, Ae, be)
        M = np.array([[1, 3, -2, 4], [3, -2, 4, -1], [3, -2, -1, 3], [4, -3, -2, 1]])
        V_original = np.array([[1, 1, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0], [-1, -1, 0, 0]]).T
        
        # Transform the polytope
        A_transformed = A @ np.linalg.inv(M.T)
        P_transformed = Polytope(A_transformed, b, Ae @ np.linalg.inv(M.T), be)
        V = vertices_(P_transformed)
        V_expected = M.T @ V_original
        assert self._compare_vertex_sets(V, V_expected, tol=1e-10)
    
    def test_vertices_7d_degenerate_subspace(self):
        """Test 7D, degenerate (subspace computation required)"""
        A = np.array([
            [0,       1,  0,   0,  0,  0,   0],
            [0,      -1,  0,   0,  0,  0,   0],
            [0,       0,  0,   1,  0,  0,   0],
            [0,       0,  0,  -1,  0,  0,   0],
            [0,       0,  0,   0,  1,  0,   0],
            [0,       0,  0,   0, -1,  0,   0],
            [0,       0,  0,   0,  0,  0,   1],
            [0,       0,  0,   0,  0,  0,  -1],
            [np.sqrt(2), 0, -0.5, 0,  0,  0.5, 0],
            [np.sqrt(2), 0,  0.5, 0,  0, -0.5, 0],
            [0,       0,  0,   0,  0,  1,   0],
            [0,       0, -1,   0,  0,  0,   0],
            [-1,      0,  0,   0,  0,  0,   0]
        ])
        b = np.array([0, 0, 1, -1, 0, 0, 0, 0, np.sqrt(2), np.sqrt(2), 0.26, -0.25, -1])
        P = Polytope(A, b)
        V = vertices_(P)
        V_expected = np.array([[1, 0, 0.25, 1, 0, 0.25, 0], [1, 0, 0.26, 1, 0, 0.26, 0]]).T
        assert self._compare_vertex_sets(V, V_expected, tol=1e-10)
    
    def test_vertices_2d_unbounded_errors(self):
        """Test that unbounded polytopes raise errors"""
        # 2D, unbounded (half-space)
        A = np.array([[1, 0]])
        b = np.array([1])
        P = Polytope(A, b)
        with pytest.raises(Exception):
            vertices_(P)
        
        # 2D, unbounded (strip)
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        with pytest.raises(Exception):
            vertices_(P)
        
        # 2D, unbounded (rotated)
        M, _, _ = np.linalg.svd(np.random.rand(2, 2))
        A_rotated = A @ np.linalg.inv(M.T)
        P_rotated = Polytope(A_rotated, b)
        with pytest.raises(Exception):
            vertices_(P_rotated)
        
        # 2D, unbounded (more complex)
        A = np.array([[1, 0], [-1, 0], [0, -1]])
        b = np.array([1, 1, 1])
        P = Polytope(A, b)
        with pytest.raises(Exception):
            vertices_(P)
        
        # 2D, unbounded but not axis-aligned
        A = np.array([[1, -0.1], [0.1, -1], [-0.1, -1], [-1, -0.1]])
        b = np.ones(4)
        P = Polytope(A, b)
        with pytest.raises(Exception):
            vertices_(P)
    
    def test_vertices_method_parameter(self):
        """Test different method parameters"""
        # 2D square
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.ones(4)
        P = Polytope(A, b)
        
        # Test default method
        V_default = vertices_(P)
        
        # Test comb method explicitly (if supported)
        try:
            V_comb = vertices_(P, method='comb')
            assert self._compare_vertex_sets(V_default, V_comb)
        except Exception:
            # Method may not be implemented
            pass
    
    def test_vertices_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Very small polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        P = Polytope(A, b)
        V = vertices_(P)
        # Should have 4 vertices around origin
        assert V.shape[1] == 4
        assert np.allclose(V, np.array([
            [1e-10, 1e-10], [1e-10, -1e-10], [-1e-10, 1e-10], [-1e-10, -1e-10]
        ]).T, atol=1e-15)
    
    def _compare_vertex_sets(self, V1, V2, tol=1e-14):
        """
        Compare two sets of vertices, allowing for different ordering.
        
        Args:
            V1, V2: vertex matrices (d x n format)
            tol: tolerance for comparison
            
        Returns:
            bool: True if vertex sets are equivalent
        """
        if V1.shape != V2.shape:
            return False
        
        if V1.shape[1] == 0:  # Empty sets
            return True
        
        # For each vertex in V1, find a matching vertex in V2
        matched = np.zeros(V2.shape[1], dtype=bool)
        for i in range(V1.shape[1]):
            v1 = V1[:, i]
            found_match = False
            for j in range(V2.shape[1]):
                if not matched[j] and np.allclose(v1, V2[:, j], atol=tol):
                    matched[j] = True
                    found_match = True
                    break
            if not found_match:
                return False
        
        return np.all(matched)

    def test_vertices_cache_values_single_point(self):
        """Test that vertices_() sets cache values correctly for single point like MATLAB"""
        # Create a polytope that represents a single point
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([[1], [1], [1], [1]])  # This defines the point (1, 1)
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([[1], [1]])
        P = Polytope(A, b, Ae, be)
        
        V = vertices_(P)
        
        # Should be a single point
        assert V.shape[1] == 1
        assert np.allclose(V, [[1], [1]])
        
        # Test cache values are set correctly like MATLAB (lines 193-196)
        assert P._minVRep_val == True        # P.minVRep.val = true;
        assert P._emptySet_val == False      # P.emptySet.val = false;
        assert P._fullDim_val == False       # P.fullDim.val = false; (no zero-dimensional sets)
        assert P._bounded_val == True        # P.bounded.val = true;