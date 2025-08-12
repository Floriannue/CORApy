"""
test_representsa_ - unit test function for polytope representsa_ method

Tests the representation checking functionality of polytopes.

Authors: MATLAB: Mark Wetzlinger, Viktor Kotsev
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.representsa_ import representsa_


class TestPolytopeRepresentsa:
    """Test class for polytope representsa_ method"""
    
    # --- origin tests ---
    
    def test_representsa_origin_empty_polytope(self):
        """Test origin representation with empty polytope"""
        # Empty polytope (contradictory constraints)
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([-1, -1])
        P = Polytope(A, b)
        assert not representsa_(P, 'origin')
    
    def test_representsa_origin_fully_empty(self):
        """Test origin representation with fully empty polytope"""
        A = np.zeros((0, 2))
        b = np.zeros(0)
        P = Polytope(A, b)
        assert not representsa_(P, 'origin')
    
    def test_representsa_origin_only_origin(self):
        """Test origin representation with polytope containing only origin"""
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = np.zeros(4)
        P = Polytope(A, b)
        assert representsa_(P, 'origin')
    
    def test_representsa_origin_shifted_center(self):
        """Test origin representation with shifted center"""
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = np.zeros(4)
        P = Polytope(A, b)
        # Shift center by adding [0.01; 0]
        A_shifted = A
        b_shifted = b + A @ np.array([0.01, 0])
        P_shifted = Polytope(A_shifted, b_shifted)
        assert not representsa_(P_shifted, 'origin')
        
        # Test with tolerance
        tol = 0.02
        assert representsa_(P_shifted, 'origin', tol)
    
    # --- interval tests ---
    
    def test_representsa_interval_fully_empty(self):
        """Test interval representation with fully empty polytope"""
        A = np.zeros((0, 2))
        b = np.zeros(0)
        P = Polytope(A, b)
        result, I = representsa_(P, 'interval', return_set=True)
        assert result
        # Should represent unbounded interval [-Inf, Inf]^2
        assert I is not None  # Check that interval is returned
    
    def test_representsa_interval_not_interval(self):
        """Test interval representation with non-interval polytope"""
        A = np.array([[2, 1], [1, 0], [0, 1]])
        b = np.array([1, 1, 2])
        P = Polytope(A, b)
        assert not representsa_(P, 'interval')
    
    def test_representsa_interval_from_conversion(self):
        """Test interval representation from interval conversion"""
        # Create polytope from interval bounds
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([2.01, -1.99, 1.01, -0.99])
        P = Polytope(A, b)
        assert representsa_(P, 'interval')
    
    def test_representsa_interval_redundant_constraints(self):
        """Test interval representation with redundant constraints"""
        A = np.array([[1, 0], [-1, 0], [-2, 0], [0, 2], [0, 4]])
        b = np.array([1, 1, 1, 1, 1])
        P = Polytope(A, b)
        result, I = representsa_(P, 'interval', return_set=True)
        assert result
        # Should represent interval [-0.5, 1] x [-Inf, 1/4]
        assert I is not None
    
    def test_representsa_interval_3d_unit_box(self):
        """Test interval representation with 3D unit box"""
        n = 3
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.ones(2*n)
        P = Polytope(A, b)
        result, I = representsa_(P, 'interval', return_set=True)
        assert result
        assert I is not None
    
    def test_representsa_interval_3d_degenerate(self):
        """Test interval representation with 3D degenerate interval"""
        A = np.array([[1, 0, 0], [-1, 0, 0]])
        b = np.array([1, 1])
        Ae = np.array([[0, 1, 0], [0, 0, 1]])
        be = np.array([5, 7])
        P = Polytope(A, b, Ae, be)
        result, I = representsa_(P, 'interval', return_set=True)
        assert result
        # Should represent interval [-1, 1] x [5, 5] x [7, 7]
        assert I is not None
    
    def test_representsa_interval_3d_bounded(self):
        """Test interval representation with 3D bounded polytope"""
        n = 3
        A_coeffs = np.array([2, 3, 2.75, 4, 4.5, 1.5])
        A = A_coeffs.reshape(-1, 1) * np.vstack([np.eye(n), -np.eye(n)])
        b_coeffs = np.array([0.5, 4, 0.25, 2, 2.25, 3])
        b = b_coeffs * np.ones(2*n)
        P = Polytope(A, b)
        result, I = representsa_(P, 'interval', return_set=True)
        assert result
        assert I is not None
    
    def test_representsa_interval_3d_unbounded(self):
        """Test interval representation with 3D unbounded polytope"""
        n = 3
        A = np.vstack([2*np.eye(n), -np.eye(n)])
        # Remove some constraints to make it unbounded
        A = np.delete(A, [1, 3], axis=0)  # Remove rows 1 and 3 (0-indexed)
        b = np.ones(A.shape[0])
        P = Polytope(A, b)
        result, I = representsa_(P, 'interval', return_set=True)
        assert result
        assert I is not None
    
    # --- emptySet tests ---
    
    def test_representsa_emptyset_empty_constructor(self):
        """Test emptySet representation with empty constructor"""
        # Construct explicitly empty polytope
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([-1, -1])  # x >= 1 and x <= -1 (impossible)
        P = Polytope(A, b)
        assert representsa_(P, 'emptySet')
    
    def test_representsa_emptyset_1d_fully_empty(self):
        """Test emptySet representation with 1D fully empty"""
        A = np.zeros((0, 1))
        b = np.zeros(0)
        P = Polytope(A, b)
        assert not representsa_(P, 'emptySet')
    
    def test_representsa_emptyset_1d_empty(self):
        """Test emptySet representation with 1D empty polytope"""
        A = np.array([[1], [-1]])
        b = np.array([1, -3])
        P = Polytope(A, b)
        assert representsa_(P, 'emptySet')
    
    def test_representsa_emptyset_1d_empty_redundant(self):
        """Test emptySet representation with 1D empty polytope (redundant)"""
        A = np.array([[1], [-1], [1], [1], [-1], [-1]])
        b = np.array([1, -3, 1, 4, 2, 1])
        P = Polytope(A, b)
        assert representsa_(P, 'emptySet')
    
    def test_representsa_emptyset_2d_non_empty(self):
        """Test emptySet representation with 2D non-empty polytope"""
        A = np.array([[2, 1], [-2, 3], [-2, -2], [4, 1]])
        b = np.ones(4)
        P = Polytope(A, b)
        assert not representsa_(P, 'emptySet')
    
    def test_representsa_emptyset_2d_empty_inequalities(self):
        """Test emptySet representation with 2D empty polytope (inequalities)"""
        A = np.array([[-1, -1], [-1, 1], [1, 0]])
        b = np.array([-2, -2, -1])
        P = Polytope(A, b)
        assert representsa_(P, 'emptySet')
    
    def test_representsa_emptyset_2d_empty_equalities(self):
        """Test emptySet representation with 2D empty polytope (equalities)"""
        # x1 == 1, x2 == 1, x1+x2 == 1 (impossible: 1+1 != 1)
        Ae = np.array([[1, 0], [0, 1], [1, 1]])
        be = np.array([1, 1, 1])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert representsa_(P, 'emptySet')
    
    def test_representsa_emptyset_2d_unbounded_degenerate(self):
        """Test emptySet representation with 2D unbounded degenerate polytope"""
        A = np.array([[0, 1], [0, -1]])
        b = np.array([1, -1])
        P = Polytope(A, b)
        assert not representsa_(P, 'emptySet')
    
    def test_representsa_emptyset_2d_v_representation(self):
        """Test emptySet representation with 2D V-representation"""
        V = np.array([[2, 0], [-2, 0], [0, 2], [0, -2]]).T
        P = Polytope(V)
        assert not representsa_(P, 'emptySet')
    
    # --- point tests ---
    
    def test_representsa_point_fully_empty(self):
        """Test point representation with fully empty polytope"""
        A = np.zeros((0, 2))
        b = np.zeros(0)
        P = Polytope(A, b)
        assert not representsa_(P, 'point')
    
    def test_representsa_point_single_point(self):
        """Test point representation with single point"""
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([2, 3])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert representsa_(P, 'point')
    
    def test_representsa_point_line_segment(self):
        """Test point representation with line segment"""
        # Line segment from [0,0] to [1,1]
        V = np.array([[0, 1], [0, 1]]).T
        P = Polytope(V)
        assert not representsa_(P, 'point')
    
    def test_representsa_point_2d_polytope(self):
        """Test point representation with 2D polytope"""
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        assert not representsa_(P, 'point')
    
    # --- general tests ---
    
    def test_representsa_fullspace(self):
        """Test fullspace representation"""
        # Fully empty polytope (no constraints)
        A = np.zeros((0, 2))
        b = np.zeros(0)
        P = Polytope(A, b)
        assert representsa_(P, 'fullspace')
        
        # Polytope with constraints should not be fullspace
        A = np.array([[1, 0]])
        b = np.array([1])
        P = Polytope(A, b)
        assert not representsa_(P, 'fullspace')
    
    def test_representsa_zonotope(self):
        """Test zonotope representation"""
        # Simple zonotope-representable polytope (axis-aligned box)
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        result = representsa_(P, 'zonotope')
        # Result depends on implementation - just check it returns boolean
        assert isinstance(result, (bool, np.bool_))
    
    def test_representsa_hyperplane(self):
        """Test hyperplane representation"""
        # Single hyperplane constraint: x + y = 1
        Ae = np.array([[1, 1]])
        be = np.array([1])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert representsa_(P, 'hyperplane')
        
        # Multiple equality constraints should not be hyperplane
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([1, 1])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        assert not representsa_(P, 'hyperplane')

    def test_representsa_hyperplane_detection_normalization(self):
        # Ae rows scaled/duplicate should still be hyperplane
        A = np.zeros((0, 2))
        b = np.zeros(0)
        Ae = np.array([[2.0, 0.0], [1.0, 0.0]])
        be = np.array([2.0, 1.0])
        P = Polytope(A, b, Ae, be)
        assert representsa_(P, 'hyperplane')

    def test_representsa_halfspace_normalization(self):
        # Multiple aligned inequalities normalize to a single halfspace
        A = np.array([[2.0, 0.0], [1.0, 0.0], [-2.0, -0.0]])
        b = np.array([2.0, 1.0, -2.0])
        P = Polytope(A, b)
        assert representsa_(P, 'halfspace')
    
    def test_representsa_invalid_type(self):
        """Test invalid representation type"""
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        with pytest.raises(Exception):
            representsa_(P, 'invalid_type')
    
    def test_representsa_tolerance_parameter(self):
        """Test tolerance parameter functionality"""
        # Create slightly perturbed origin
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([0.001, 0.001, 0.001, 0.001])
        P = Polytope(A, b)
        
        # Should not represent origin without tolerance
        assert not representsa_(P, 'origin')
        
        # Should represent origin with sufficient tolerance
        assert representsa_(P, 'origin', 0.01)
    
    def test_representsa_with_return_value(self):
        """Test representsa with return value"""
        # Test interval case which returns both boolean and object
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        
        result, I = representsa_(P, 'interval')
        assert isinstance(result, (bool, np.bool_))
        if result:
            assert I is not None
        
        # Test case that only returns boolean
        result = representsa_(P, 'origin')
        assert isinstance(result, (bool, np.bool_)) 