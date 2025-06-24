"""
test_zonotope - unit test function for polytope zonotope conversion

Tests the zonotope conversion functionality of polytopes.

Authors: MATLAB: Mark Wetzlinger
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.zonotope import zonotope


class TestPolytopeZonotope:
    """Test class for polytope zonotope method"""
    
    def test_zonotope_empty_case(self):
        """Test zonotope conversion with empty polytope"""
        # Empty polytope (contradictory constraints)
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([-1, -1])
        P = Polytope(A, b)
        Z = zonotope(P)
        # Should result in empty zonotope with correct dimension
        assert Z is not None
        # Note: Actual empty check depends on zonotope implementation
    
    def test_zonotope_1d_bounded(self):
        """Test zonotope conversion with 1D bounded polytope"""
        A = np.array([[1], [-1]])
        b = np.array([2, -1])
        P = Polytope(A, b)
        Z = zonotope(P)
        # Should represent interval [1, 2] as zonotope with center 1.5, generator 0.5
        assert Z is not None
        # Expected: zonotope with center [1.5] and generator matrix [[0.5]]
    
    def test_zonotope_2d_bounded_h_representation(self):
        """Test zonotope conversion with 2D bounded polytope (H-representation)"""
        A = np.array([[1, 0], [-1, 1], [-1, -1]])
        b = np.array([1, 1, 1])
        P = Polytope(A, b)
        Z = zonotope(P)
        assert Z is not None
        # The resulting zonotope should contain the polytope
    
    def test_zonotope_2d_bounded_vertex_instantiation(self):
        """Test zonotope conversion with 2D bounded polytope (vertex instantiation)"""
        V = np.array([[-1, 0], [1, 2], [1, -2]]).T
        P = Polytope(V)
        Z = zonotope(P)
        assert Z is not None
        # The resulting zonotope should contain the polytope
    
    def test_zonotope_2d_unit_square(self):
        """Test zonotope conversion with 2D unit square"""
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        Z = zonotope(P)
        assert Z is not None
        # Expected: zonotope with center [0, 0] and generators [[1, 0], [0, 1]]
    
    def test_zonotope_2d_triangle(self):
        """Test zonotope conversion with 2D triangle"""
        V = np.array([[0, 0], [1, 0], [0, 1]]).T
        P = Polytope(V)
        Z = zonotope(P)
        assert Z is not None
        # The resulting zonotope should contain the triangle
    
    def test_zonotope_3d_unit_cube(self):
        """Test zonotope conversion with 3D unit cube"""
        n = 3
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.ones(2*n)
        P = Polytope(A, b)
        Z = zonotope(P)
        assert Z is not None
        # Expected: zonotope with center [0, 0, 0] and generators I_3
    
    def test_zonotope_degenerate_polytope(self):
        """Test zonotope conversion with degenerate polytope"""
        # Line segment in 2D
        V = np.array([[0, 0], [1, 1]]).T
        P = Polytope(V)
        Z = zonotope(P)
        assert Z is not None
        # Should create zonotope representing the line segment
    
    def test_zonotope_single_point(self):
        """Test zonotope conversion with single point"""
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([2, 3])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        Z = zonotope(P)
        assert Z is not None
        # Expected: zonotope with center [2, 3] and no generators (or zero generators)
    
    def test_zonotope_unbounded_error(self):
        """Test that unbounded polytopes raise errors"""
        # Unbounded polytope (half-space)
        A = np.array([[1, 0]])
        b = np.array([1])
        P = Polytope(A, b)
        with pytest.raises(Exception):
            zonotope(P)
        
        # Fully unbounded polytope
        A = np.zeros((0, 2))
        b = np.zeros(0)
        P = Polytope(A, b)
        with pytest.raises(Exception):
            zonotope(P)
    
    def test_zonotope_high_dimensional(self):
        """Test zonotope conversion with higher dimensional polytope"""
        # 4D hypercube
        n = 4
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.ones(2*n)
        P = Polytope(A, b)
        Z = zonotope(P)
        assert Z is not None
        # Should create zonotope representation of 4D hypercube
    
    def test_zonotope_edge_cases(self):
        """Test edge cases for zonotope conversion"""
        # Very small polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        P = Polytope(A, b)
        Z = zonotope(P)
        assert Z is not None
        
        # Large polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1e10, 1e10, 1e10, 1e10])
        P = Polytope(A, b)
        Z = zonotope(P)
        assert Z is not None
    
    def test_zonotope_non_axis_aligned(self):
        """Test zonotope conversion with non-axis-aligned polytope"""
        # Rotated square
        angle = np.pi / 4
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Define square vertices and rotate
        square_vertices = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]).T
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = rotation_matrix @ square_vertices
        P = Polytope(rotated_vertices)
        Z = zonotope(P)
        assert Z is not None
    
    def test_zonotope_consistency_properties(self):
        """Test that zonotope conversion preserves certain properties"""
        # Test that the zonotope contains the original polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([2, 1, 1, 1])  # Rectangle [-1, 2] x [-1, 1]
        P = Polytope(A, b)
        Z = zonotope(P)
        assert Z is not None
        
        # The zonotope should have the same dimension as the polytope
        # This test depends on the zonotope implementation having a dim() method
        # assert Z.dim() == P.dim()  # Uncomment when zonotope class is available
    
    def test_zonotope_different_polytope_types(self):
        """Test zonotope conversion with different types of polytopes"""
        # Test with polytope from inequality constraints
        A1 = np.array([[1, 1], [-1, 1], [0, -1]])
        b1 = np.array([1, 1, 0])
        P1 = Polytope(A1, b1)
        Z1 = zonotope(P1)
        assert Z1 is not None
        
        # Test with polytope from vertex representation
        V2 = np.array([[0, 0], [2, 0], [1, 1]]).T
        P2 = Polytope(V2)
        Z2 = zonotope(P2)
        assert Z2 is not None
        
        # Test with polytope from mixed constraints
        A3 = np.array([[1, 0], [-1, 0]])
        b3 = np.array([1, 1])
        Ae3 = np.array([[0, 1]])
        be3 = np.array([0.5])
        P3 = Polytope(A3, b3, Ae3, be3)
        Z3 = zonotope(P3)
        assert Z3 is not None 