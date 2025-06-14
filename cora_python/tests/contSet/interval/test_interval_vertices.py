# test_interval_vertices - unit test function of vertices
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/interval/test_interval_vertices.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval


class TestIntervalVertices:
    """Test class for interval vertices method."""

    def test_interval_vertices_empty(self):
        """Test vertices of empty interval."""
        I = Interval.empty(2)
        V = I.vertices()
        assert V.shape == (2, 0)

    def test_interval_vertices_1d(self):
        """Test vertices of 1D interval."""
        I = Interval(np.array([[-2]]), np.array([[3]]))
        V = I.vertices()
        V_true = np.array([[-2, 3]]).T
        assert np.allclose(V, V_true)

    def test_interval_vertices_2d(self):
        """Test vertices of 2D interval."""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        V = I.vertices()
        # Vertices should be: [-2,1], [3,1], [3,4], [-2,4]
        V_expected = np.array([
            [-2, 3, 3, -2],
            [1, 1, 4, 4]
        ])
        # Check that all expected vertices are present
        assert V.shape[1] == 4
        for i in range(V_expected.shape[1]):
            found = False
            for j in range(V.shape[1]):
                if np.allclose(V[:, j], V_expected[:, i]):
                    found = True
                    break
            assert found, f"Vertex {V_expected[:, i]} not found in result"

    def test_interval_vertices_3d(self):
        """Test vertices of 3D interval."""
        I = Interval(
            np.array([[-1], [0], [-2]]),
            np.array([[1], [2], [1]])
        )
        V = I.vertices()
        # 3D box should have 2^3 = 8 vertices
        assert V.shape == (3, 8)
        
        # Check that all vertices are within bounds
        assert np.all(V[0, :] >= -1) and np.all(V[0, :] <= 1)
        assert np.all(V[1, :] >= 0) and np.all(V[1, :] <= 2)
        assert np.all(V[2, :] >= -2) and np.all(V[2, :] <= 1)

    def test_interval_vertices_point(self):
        """Test vertices of point interval."""
        point = np.array([[1], [2], [3]])
        I = Interval(point)
        V = I.vertices()
        assert V.shape == (3, 1)
        assert np.allclose(V, point)

    def test_interval_vertices_degenerate(self):
        """Test vertices of degenerate interval (some dimensions are points)."""
        I = Interval(
            np.array([[-1], [2], [-2]]),  # y is fixed at 2
            np.array([[1], [2], [1]])
        )
        V = I.vertices()
        # Should have 4 vertices (2^2 for x and z, y is fixed)
        assert V.shape == (3, 4)
        # All vertices should have y = 2
        assert np.all(V[1, :] == 2)

    def test_interval_vertices_unbounded(self):
        """Test vertices of unbounded interval should raise error or handle gracefully."""
        I = Interval(
            np.array([[-np.inf], [-1]]),
            np.array([[1], [1]])
        )
        # Unbounded intervals should handle vertices appropriately
        # This might raise an error or return a special result
        try:
            V = I.vertices()
            # If it doesn't raise an error, check that finite bounds are respected
            assert np.all(V[1, :] >= -1) and np.all(V[1, :] <= 1)
        except (ValueError, RuntimeError):
            # It's acceptable for unbounded intervals to raise an error
            pass

    def test_interval_vertices_large_dimension(self):
        """Test vertices of higher dimensional interval."""
        n = 4
        I = Interval(-np.ones((n, 1)), np.ones((n, 1)))
        V = I.vertices()
        # Should have 2^n vertices
        assert V.shape == (n, 2**n)
        # All vertices should be within [-1, 1]^n
        assert np.all(V >= -1) and np.all(V <= 1)

    def test_interval_vertices_matrix(self):
        """Test vertices of matrix interval."""
        I = Interval(
            np.array([[-1, 0], [1, -2]]),
            np.array([[2, 1], [3, 0]])
        )
        V = I.vertices()
        # Matrix intervals should be handled appropriately
        # Check that dimensions are correct
        assert V.shape[0] == I.inf.size
        # Check that vertices respect bounds
        I_flat = I.inf.flatten()
        U_flat = I.sup.flatten()
        for i in range(V.shape[1]):
            v_reshaped = V[:, i]
            assert np.all(v_reshaped >= I_flat) and np.all(v_reshaped <= U_flat)


def test_interval_vertices():
    """Test function for interval vertices method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestIntervalVertices()
    test.test_interval_vertices_empty()
    test.test_interval_vertices_1d()
    test.test_interval_vertices_2d()
    test.test_interval_vertices_3d()
    test.test_interval_vertices_point()
    test.test_interval_vertices_degenerate()
    test.test_interval_vertices_unbounded()
    test.test_interval_vertices_large_dimension()
    test.test_interval_vertices_matrix()
    
    print("test_interval_vertices: all tests passed")


if __name__ == "__main__":
    test_interval_vertices() 