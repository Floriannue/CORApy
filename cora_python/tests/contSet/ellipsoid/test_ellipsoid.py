import unittest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestEllipsoid(unittest.TestCase):
    """Test cases for Ellipsoid class constructor and basic methods."""

    def test_constructor_basic(self):
        """Test basic ellipsoid constructor with Q matrix only."""
        Q = np.array([[2.7, -0.2], [-0.2, 2.4]])
        E = Ellipsoid(Q)
        
        self.assertTrue(np.allclose(E.Q, Q))
        self.assertTrue(np.allclose(E.q, np.zeros((2, 1))))
        self.assertEqual(E.TOL, 1e-6)
        self.assertEqual(E.precedence, 50)

    def test_constructor_with_center(self):
        """Test ellipsoid constructor with Q matrix and center."""
        Q = np.array([[2.7, -0.2], [-0.2, 2.4]])
        q = np.array([[1], [2]])
        E = Ellipsoid(Q, q)
        
        self.assertTrue(np.allclose(E.Q, Q))
        self.assertTrue(np.allclose(E.q, q))
        self.assertEqual(E.TOL, 1e-6)

    def test_constructor_with_tolerance(self):
        """Test ellipsoid constructor with Q, q, and tolerance."""
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([[0], [0]])
        TOL = 1e-8
        E = Ellipsoid(Q, q, TOL)
        
        self.assertTrue(np.allclose(E.Q, Q))
        self.assertTrue(np.allclose(E.q, q))
        self.assertEqual(E.TOL, TOL)

    def test_copy_constructor(self):
        """Test copy constructor."""
        Q = np.array([[2.0, 0.5], [0.5, 3.0]])
        q = np.array([[1], [-1]])
        E1 = Ellipsoid(Q, q)
        E2 = Ellipsoid(E1)
        
        self.assertTrue(np.allclose(E2.Q, E1.Q))
        self.assertTrue(np.allclose(E2.q, E1.q))
        self.assertEqual(E2.TOL, E1.TOL)
        self.assertEqual(E2.precedence, E1.precedence)

    def test_constructor_empty_input(self):
        """Test that empty constructor raises error."""
        with self.assertRaises(Exception):  # CORAerror raises generic Exception
            Ellipsoid()

    def test_constructor_invalid_dimensions(self):
        """Test constructor with mismatched dimensions."""
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([[1], [2], [3]])  # Wrong dimension
        
        with self.assertRaises(Exception):  # CORAerror raises generic Exception
            Ellipsoid(Q, q)

    def test_constructor_non_square_matrix(self):
        """Test constructor with non-square matrix."""
        Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Non-square
        
        with self.assertRaises(Exception):  # CORAerror raises generic Exception
            Ellipsoid(Q)

    def test_constructor_non_symmetric_matrix(self):
        """Test constructor with non-symmetric matrix."""
        Q = np.array([[1.0, 0.5], [0.3, 1.0]])  # Non-symmetric
        
        with self.assertRaises(Exception):  # CORAerror raises generic Exception
            Ellipsoid(Q)

    def test_constructor_negative_definite_matrix(self):
        """Test constructor with negative definite matrix."""
        Q = np.array([[-1.0, 0.0], [0.0, -1.0]])  # Negative definite
        
        with self.assertRaises(Exception):  # CORAerror raises generic Exception
            Ellipsoid(Q)

    def test_constructor_1d_ellipsoid(self):
        """Test constructor for 1D ellipsoid."""
        Q = np.array([[4.0]])
        q = np.array([[2.0]])
        E = Ellipsoid(Q, q)
        
        self.assertTrue(np.allclose(E.Q, Q))
        self.assertTrue(np.allclose(E.q, q))

    def test_constructor_3d_ellipsoid(self):
        """Test constructor for 3D ellipsoid."""
        Q = np.array([[2.0, 0.0, 0.0], 
                      [0.0, 3.0, 0.0], 
                      [0.0, 0.0, 1.0]])
        q = np.array([[1], [2], [3]])
        E = Ellipsoid(Q, q)
        
        self.assertTrue(np.allclose(E.Q, Q))
        self.assertTrue(np.allclose(E.q, q))

    def test_constructor_zero_matrix(self):
        """Test constructor with zero matrix (degenerate case)."""
        Q = np.array([[0.0, 0.0], [0.0, 0.0]])
        q = np.array([[0], [0]])
        E = Ellipsoid(Q, q)
        
        self.assertTrue(np.allclose(E.Q, Q))
        self.assertTrue(np.allclose(E.q, q))

    def test_constructor_empty_matrices(self):
        """Test constructor with empty matrices."""
        Q = np.array([]).reshape(0, 0)
        q = np.array([]).reshape(0, 0)
        E = Ellipsoid(Q, q)
        
        self.assertEqual(E.Q.shape, (0, 0))
        self.assertEqual(E.q.shape, (0, 0))


if __name__ == '__main__':
    unittest.main() 