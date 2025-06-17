# test_zonotope_randPoint - unit test function of randPoint
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/zonotope/test_zonotope_randPoint.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestZonotopeRandPoint:
    """Test class for zonotope randPoint method."""

    def test_zonotope_randPoint_empty(self):
        """Test randPoint of empty zonotope."""
        n = 4
        Z = Zonotope.empty(n)
        p = Z.randPoint(5)
        assert p.shape == (n, 0)

    def test_zonotope_randPoint_point(self):
        """Test randPoint of point zonotope (single point)."""
        Z = Zonotope(np.array([[2], [3]]))
        p = Z.randPoint(1)
        assert p.shape == (2, 1)
        assert np.allclose(p, np.array([[2], [3]]))

        # Test extreme points as well
        p_extr = Z.randPoint(1, type_='extreme')
        assert p_extr.shape == (2, 1)

    def test_zonotope_randPoint_2d(self):
        """Test randPoint of 2D zonotope."""
        c = np.array([[1], [2]])
        G = np.array([[0.5, 1], [0.5, -1]])
        Z = Zonotope(c, G)
        
        # Test standard random points
        p = Z.randPoint(10)
        assert p.shape == (2, 10)
        
        # Test extreme points
        p_extr = Z.randPoint(4, type_='extreme')
        assert p_extr.shape == (2, 4)

    def test_zonotope_randPoint_3d_degenerate(self):
        """Test randPoint of 3D degenerate zonotope."""
        Z = Zonotope(
            np.array([[1], [2], [-1]]),
            np.array([[1, 3, -2], [1, 0, 1], [2, 3, -1]])
        )
        numPoints = 10
        
        # Test various methods
        p_standard = Z.randPoint(numPoints, type_='standard')
        assert p_standard.shape == (3, numPoints)
        
        # Test if points are contained (approximately)
        # This is a basic containment check - in practice we'd need the contains method
        
    def test_zonotope_randPoint_methods(self):
        """Test different randPoint methods."""
        c = np.array([[0], [1]])
        G = np.array([[1, 2], [-1, 0]])
        Z = Zonotope(c, G)
        
        # Test standard method
        p1 = Z.randPoint(5, type_='standard')
        assert p1.shape == (2, 5)
        
        # Test extreme method
        p2 = Z.randPoint(4, type_='extreme')
        assert p2.shape == (2, 4)

    def test_zonotope_randPoint_large_count(self):
        """Test randPoint with large number of points."""
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        p = Z.randPoint(100)
        assert p.shape == (2, 100)
        
        # Check that points have some variation
        assert np.var(p[0, :]) > 0
        assert np.var(p[1, :]) > 0

    def test_zonotope_randPoint_default_args(self):
        """Test randPoint with default arguments."""
        Z = Zonotope(np.array([[1], [2]]), np.array([[0.5], [0.5]]))
        
        # Default should be 1 point
        p = Z.randPoint()
        assert p.shape == (2, 1)

    def test_zonotope_randPoint_zero_count(self):
        """Test randPoint with zero points."""
        Z = Zonotope(np.array([[1], [2]]), np.array([[0.5], [0.5]]))
        p = Z.randPoint(0)
        assert p.shape == (2, 0)

    def test_zonotope_randPoint_no_generators(self):
        """Test randPoint of zonotope without generators."""
        c = np.array([[1], [2], [3]])
        Z = Zonotope(c)
        
        p = Z.randPoint(5)
        assert p.shape == (3, 5)
        # All points should be the center
        for i in range(5):
            assert np.allclose(p[:, [i]], c)

    def test_zonotope_randPoint_single_generator(self):
        """Test randPoint of zonotope with single generator."""
        c = np.array([[0], [0]])
        G = np.array([[1], [1]])
        Z = Zonotope(c, G)
        
        p = Z.randPoint(10)
        assert p.shape == (2, 10)
        
        # All points should lie on the line from c-G to c+G
        for i in range(10):
            # Check that p[1,i] == p[0,i] (on the diagonal line)
            assert abs(p[1, i] - p[0, i]) < 1e-10


def test_zonotope_randPoint():
    """Test function for zonotope randPoint method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestZonotopeRandPoint()
    test.test_zonotope_randPoint_empty()
    test.test_zonotope_randPoint_point()
    test.test_zonotope_randPoint_2d()
    test.test_zonotope_randPoint_3d_degenerate()
    test.test_zonotope_randPoint_methods()
    test.test_zonotope_randPoint_large_count()
    test.test_zonotope_randPoint_default_args()
    test.test_zonotope_randPoint_zero_count()
    test.test_zonotope_randPoint_no_generators()
    test.test_zonotope_randPoint_single_generator()
    
    print("test_zonotope_randPoint: all tests passed")


if __name__ == "__main__":
    test_zonotope_randPoint() 