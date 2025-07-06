import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestZonotopeEllipsoid:
    """Test cases for Zonotope.ellipsoid method."""
    
    def test_ellipsoid_point(self):
        """Test conversion of point zonotope to ellipsoid."""
        # Create a point zonotope
        c = np.array([[1], [2]])
        Z = Zonotope(c)
        
        E = Z.ellipsoid()
        
        # Should be an ellipsoid with zero shape matrix at the point
        assert isinstance(E, Ellipsoid)
        np.testing.assert_array_almost_equal(E.q.flatten(), c.flatten())
        np.testing.assert_array_almost_equal(E.Q, np.zeros((2, 2)))
    
    def test_ellipsoid_parallelotope_outer(self):
        """Test conversion of parallelotope (square zonotope) to ellipsoid."""
        # Create a 2D parallelotope (square)
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(np.hstack([c, G]))
        
        E = Z.ellipsoid('outer:norm_bnd')
        
        # For parallelotope, should use fac * G * G'
        expected_Q = 2 * (G @ G.T)  # fac = n = 2 for outer
        
        assert isinstance(E, Ellipsoid)
        np.testing.assert_array_almost_equal(E.q.flatten(), c.flatten())
        np.testing.assert_array_almost_equal(E.Q, expected_Q)
    
    def test_ellipsoid_parallelotope_inner(self):
        """Test inner conversion of parallelotope to ellipsoid."""
        # Create a 2D parallelotope
        c = np.array([[1], [1]])
        G = np.array([[2, 0], [0, 3]])
        Z = Zonotope(np.hstack([c, G]))
        
        E = Z.ellipsoid('inner:norm')
        
        # For inner parallelotope, should use fac = 1
        expected_Q = G @ G.T
        
        assert isinstance(E, Ellipsoid)
        np.testing.assert_array_almost_equal(E.q.flatten(), c.flatten())
        np.testing.assert_array_almost_equal(E.Q, expected_Q)
    
    def test_ellipsoid_general_zonotope(self):
        """Test conversion of general zonotope to ellipsoid."""
        # Create a general zonotope with more generators than dimensions
        c = np.array([[1], [-1]])
        G = np.array([[2, -4, 3], [3, 2, -4]])
        Z = Zonotope(np.hstack([c, G]))
        
        E = Z.ellipsoid('outer:norm_bnd')
        
        # Should create an ellipsoid using norm-based approximation
        assert isinstance(E, Ellipsoid)
        np.testing.assert_array_almost_equal(E.q.flatten(), c.flatten())
        # Shape matrix should be positive definite
        eigenvals = np.linalg.eigvals(E.Q)
        assert np.all(eigenvals >= 0)
    
    def test_ellipsoid_exact_modes(self):
        """Test that exact modes work correctly."""
        c = np.array([[0], [0]])
        G = np.array([[1, 2, 3], [4, 5, 6]])
        Z = Zonotope(np.hstack([c, G]))
        
        # Test outer:exact
        E_outer_exact = Z.ellipsoid('outer:exact')
        assert isinstance(E_outer_exact, Ellipsoid)
        
        # Test outer:norm
        E_outer_norm = Z.ellipsoid('outer:norm')
        assert isinstance(E_outer_norm, Ellipsoid)
        
        # Test inner:exact
        E_inner_exact = Z.ellipsoid('inner:exact')
        assert isinstance(E_inner_exact, Ellipsoid)
        
        # Test inner:norm
        E_inner_norm = Z.ellipsoid('inner:norm')
        assert isinstance(E_inner_norm, Ellipsoid)
    
    def test_ellipsoid_invalid_mode(self):
        """Test invalid mode parameter."""
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(np.hstack([c, G]))
        
        with pytest.raises(CORAerror) as exc_info:
            Z.ellipsoid('invalid_mode')
        
        assert exc_info.value.id == 'CORA:wrongValue'
    
    def test_ellipsoid_degenerate_zonotope(self):
        """Test conversion of degenerate (non-full-dimensional) zonotope."""
        # Create a degenerate zonotope (line in 2D)
        c = np.array([[1], [2]])
        G = np.array([[1], [1]]) # Single generator - creates a line
        Z = Zonotope(np.hstack([c, G]))
        
        # This should work but might have special handling
        E = Z.ellipsoid()
        
        assert isinstance(E, Ellipsoid)
        # The result should be valid even for degenerate case
        np.testing.assert_array_almost_equal(E.q.flatten(), c.flatten()) 