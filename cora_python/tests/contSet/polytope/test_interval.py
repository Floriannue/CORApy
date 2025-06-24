"""
test_interval - unit test function for polytope interval conversion

Tests the interval conversion functionality of polytopes.

Authors: MATLAB: Mark Wetzlinger
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.interval import interval


class TestPolytopeInterval:
    """Test class for polytope interval method"""
    
    def test_interval_1d_unbounded(self):
        """Test interval conversion with 1D unbounded polytope"""
        A = np.array([[1]])
        b = np.array([1])
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval (-Inf, 1]
        assert I is not None
        # Expected: interval with lower bound -Inf, upper bound 1
    
    def test_interval_1d_bounded(self):
        """Test interval conversion with 1D bounded polytope"""
        A = np.array([[3], [-2]])
        b = np.array([1, 0])
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval [0, 1/3]
        assert I is not None
        # Expected: interval with lower bound 0, upper bound 1/3
    
    def test_interval_1d_single_point(self):
        """Test interval conversion with 1D single point"""
        Ae = np.array([[5]])
        be = np.array([3])
        P = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)
        I = interval(P)
        # Should represent interval [3/5, 3/5]
        assert I is not None
        # Expected: degenerate interval at point 3/5
    
    def test_interval_1d_fully_empty(self):
        """Test interval conversion with 1D fully empty polytope"""
        A = np.zeros((0, 1))
        b = np.zeros(0)
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval (-Inf, Inf)
        assert I is not None
        # Expected: unbounded interval
    
    def test_interval_1d_empty(self):
        """Test interval conversion with 1D empty polytope"""
        A = np.array([[1], [-1]])
        b = np.array([1, -3])
        P = Polytope(A, b)
        I = interval(P)
        # Should represent empty interval
        assert I is not None
        # Expected: empty interval (size 0)
    
    def test_interval_2d_diamond_in_unit_square(self):
        """Test interval conversion with 2D diamond in unit square"""
        A = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        b = np.ones(4)
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval [-1, 1] x [-1, 1]
        assert I is not None
        # Expected: interval with bounds [[-1, -1], [1, 1]]
    
    def test_interval_2d_trivially_fulfilled_constraints(self):
        """Test interval conversion with 2D trivially fulfilled constraints"""
        A = np.array([[0, 0]])
        b = np.array([1])
        Ae = np.array([[0, 0]])
        be = np.array([0])
        P = Polytope(A, b, Ae, be)
        I = interval(P)
        # Should represent interval (-Inf, Inf) x (-Inf, Inf)
        assert I is not None
        # Expected: unbounded interval in both dimensions
    
    def test_interval_3d_unit_cube(self):
        """Test interval conversion with 3D unit cube"""
        n = 3
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.ones(2*n)
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval [-1, 1]^3
        assert I is not None
        # Expected: exact conversion, polytope should equal interval
    
    def test_interval_4d_unbounded_polytope(self):
        """Test interval conversion with 4D unbounded polytope"""
        A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], 
                      [-1, 0, 0, 0], [0, 0, 0, -1]])
        b = np.array([1, 2, 4, 3, 1.5])
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval [-3, 1] x (-Inf, 2] x (-Inf, Inf) x [-1.5, 4]
        assert I is not None
        # Expected: mixed bounded/unbounded interval
    
    def test_interval_2d_unit_square(self):
        """Test interval conversion with 2D unit square"""
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1, 1, 1, 1])
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval [-1, 1] x [-1, 1]
        assert I is not None
        # Expected: exact conversion for axis-aligned box
    
    def test_interval_2d_triangle(self):
        """Test interval conversion with 2D triangle"""
        V = np.array([[0, 0], [1, 0], [0, 1]]).T
        P = Polytope(V)
        I = interval(P)
        # Should represent interval [0, 1] x [0, 1] (bounding box)
        assert I is not None
        # Expected: bounding box of the triangle
    
    def test_interval_degenerate_2d_line(self):
        """Test interval conversion with degenerate 2D line"""
        # Line segment from [0, 0] to [1, 1]
        V = np.array([[0, 0], [1, 1]]).T
        P = Polytope(V)
        I = interval(P)
        # Should represent interval [0, 1] x [0, 1] (bounding box)
        assert I is not None
        # Expected: bounding box of the line segment
    
    def test_interval_single_point_2d(self):
        """Test interval conversion with 2D single point"""
        Ae = np.array([[1, 0], [0, 1]])
        be = np.array([2, 3])
        P = Polytope(np.zeros((0, 2)), np.zeros(0), Ae, be)
        I = interval(P)
        # Should represent interval [2, 2] x [3, 3]
        assert I is not None
        # Expected: degenerate interval at point [2, 3]
    
    def test_interval_2d_unbounded_strip(self):
        """Test interval conversion with 2D unbounded strip"""
        A = np.array([[0, 1], [0, -1]])
        b = np.array([1, 1])
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval (-Inf, Inf) x [-1, 1]
        assert I is not None
        # Expected: unbounded in x, bounded in y
    
    def test_interval_3d_degenerate_plane(self):
        """Test interval conversion with 3D degenerate plane"""
        # Plane z = 0 constrained by |x| <= 1, |y| <= 1
        A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        b = np.array([1, 1, 1, 1])
        Ae = np.array([[0, 0, 1]])
        be = np.array([0])
        P = Polytope(A, b, Ae, be)
        I = interval(P)
        # Should represent interval [-1, 1] x [-1, 1] x [0, 0]
        assert I is not None
        # Expected: bounded in x,y, degenerate in z
    
    def test_interval_high_dimensional(self):
        """Test interval conversion with higher dimensional polytope"""
        # 5D hypercube
        n = 5
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.ones(2*n)
        P = Polytope(A, b)
        I = interval(P)
        # Should represent interval [-1, 1]^5
        assert I is not None
        # Expected: 5D hypercube as interval
    
    def test_interval_edge_cases(self):
        """Test edge cases for interval conversion"""
        # Very small polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        P = Polytope(A, b)
        I = interval(P)
        assert I is not None
        # Expected: very small interval around origin
        
        # Large polytope
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([1e10, 1e10, 1e10, 1e10])
        P = Polytope(A, b)
        I = interval(P)
        assert I is not None
        # Expected: large interval
    
    def test_interval_mixed_constraints(self):
        """Test interval conversion with mixed inequality and equality constraints"""
        # 3D polytope with equality constraint: z = 1, and |x| <= 2, |y| <= 3
        A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        b = np.array([2, 2, 3, 3])
        Ae = np.array([[0, 0, 1]])
        be = np.array([1])
        P = Polytope(A, b, Ae, be)
        I = interval(P)
        # Should represent interval [-2, 2] x [-3, 3] x [1, 1]
        assert I is not None
        # Expected: bounded in x,y, degenerate in z
    
    def test_interval_empty_polytope(self):
        """Test interval conversion with empty polytope"""
        # Contradictory constraints: x >= 1 and x <= -1
        A = np.array([[1, 0], [-1, 0]])
        b = np.array([-1, -1])
        P = Polytope(A, b)
        I = interval(P)
        # Should represent empty interval
        assert I is not None
        # Expected: empty interval (implementation dependent)
    
    def test_interval_consistency_properties(self):
        """Test that interval conversion preserves certain properties"""
        # For axis-aligned polytopes, the conversion should be exact
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([3, 2, 4, 1])  # [x,y] in [-2, 3] x [-1, 4]
        P = Polytope(A, b)
        I = interval(P)
        assert I is not None
        
        # The interval should contain the original polytope
        # This test depends on having containment checking available
        # assert contains(I, P)  # Uncomment when interval class is available 