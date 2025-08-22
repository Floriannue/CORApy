"""
Test for nnHelper.compBoundsPolyZono function

This test verifies that the compBoundsPolyZono function works correctly for computing bounds of polynomial zonotopes.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.compBoundsPolyZono import compBoundsPolyZono


class TestCompBoundsPolyZono:
    """Test class for compBoundsPolyZono function"""
    
    def test_compBoundsPolyZono_basic(self):
        """Test basic compBoundsPolyZono functionality"""
        # Test with simple polynomial zonotope
        c = np.array([1.0])  # center
        G = np.array([[1.0, 0.5]])  # generators
        GI = np.array([[0.2]])  # independent generators
        E = np.array([[1, 0], [0, 1]])  # exponents
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        
        # Check that result is tuple
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        
        # Check that bounds are numpy arrays
        assert isinstance(bounds[0], np.ndarray)
        assert isinstance(bounds[1], np.ndarray)
        
        # Check that lower bound <= upper bound
        assert np.all(bounds[0] <= bounds[1])
    
    def test_compBoundsPolyZono_no_independent_generators(self):
        """Test compBoundsPolyZono with no independent generators"""
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([]).reshape(1, 0)  # Empty independent generators
        E = np.array([[1, 0], [0, 1]])
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        
        # Check result structure
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        assert np.all(bounds[0] <= bounds[1])
    
    def test_compBoundsPolyZono_different_dimensions(self):
        """Test compBoundsPolyZono with different dimensions"""
        # Test 1D
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        bounds_1d = compBoundsPolyZono(c, G, GI, E)
        assert bounds_1d[0].shape == (1,)
        assert bounds_1d[1].shape == (1,)
        
        # Test 2D
        c = np.array([1.0, 2.0])
        G = np.array([[1.0, 0.5], [0.3, 0.8]])
        GI = np.array([[0.2], [0.1]])
        E = np.array([[1, 0], [0, 1]])
        
        bounds_2d = compBoundsPolyZono(c, G, GI, E)
        assert bounds_2d[0].shape == (2,)
        assert bounds_2d[1].shape == (2,)
    
    def test_compBoundsPolyZono_different_orders(self):
        """Test compBoundsPolyZono with different polynomial orders"""
        c = np.array([1.0])
        G = np.array([[1.0, 0.5, 0.3]])
        GI = np.array([[0.2]])
        
        # Test order 1
        E1 = np.array([[1, 0], [0, 1]])
        bounds1 = compBoundsPolyZono(c, G, GI, E1)
        assert np.all(bounds1[0] <= bounds1[1])
        
        # Test order 2
        E2 = np.array([[1, 0, 2], [0, 1, 1]])
        bounds2 = compBoundsPolyZono(c, G, GI, E2)
        assert np.all(bounds2[0] <= bounds2[1])
        
        # Test order 3
        E3 = np.array([[1, 0, 2, 3], [0, 1, 1, 0]])
        bounds3 = compBoundsPolyZono(c, G, GI, E3)
        assert np.all(bounds3[0] <= bounds3[1])
    
    def test_compBoundsPolyZono_edge_cases(self):
        """Test compBoundsPolyZono edge cases"""
        # Test with zero center
        c = np.array([0.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        assert np.all(bounds[0] <= bounds[1])
        
        # Test with empty generators
        c = np.array([1.0])
        G = np.array([]).reshape(1, 0)
        GI = np.array([]).reshape(1, 0)
        E = np.array([]).reshape(2, 0)
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        assert np.all(bounds[0] <= bounds[1])
        
        # Test with very small values
        c = np.array([1e-10])
        G = np.array([[1e-10, 1e-10]])
        GI = np.array([[1e-10]])
        E = np.array([[1, 0], [0, 1]])
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        assert np.all(bounds[0] <= bounds[1])
        assert np.all(np.isfinite(bounds[0]))
        assert np.all(np.isfinite(bounds[1]))
    
    def test_compBoundsPolyZono_accuracy(self):
        """Test accuracy of bounds computation"""
        # Test with known polynomial zonotope
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        
        # For this simple case, bounds should be reasonable
        # Lower bound should be <= center
        assert np.all(bounds[0] <= c)
        
        # Upper bound should be >= center
        assert np.all(bounds[1] >= c)
        
        # Bounds should be finite
        assert np.all(np.isfinite(bounds[0]))
        assert np.all(np.isfinite(bounds[1]))
    
    def test_compBoundsPolyZono_consistency(self):
        """Test that compBoundsPolyZono produces consistent results"""
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        # Call multiple times
        bounds1 = compBoundsPolyZono(c, G, GI, E)
        bounds2 = compBoundsPolyZono(c, G, GI, E)
        
        # Should be consistent
        assert np.allclose(bounds1[0], bounds2[0], atol=1e-10)
        assert np.allclose(bounds1[1], bounds2[1], atol=1e-10)
    
    def test_compBoundsPolyZono_error_handling(self):
        """Test compBoundsPolyZono error handling"""
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        # Test with mismatched dimensions
        with pytest.raises(ValueError):
            compBoundsPolyZono(c, G, GI, E[:, :1])  # Wrong number of exponents
        
        # Test with invalid center
        with pytest.raises(ValueError):
            compBoundsPolyZono(None, G, GI, E)
    
    def test_compBoundsPolyZono_numerical_stability(self):
        """Test numerical stability"""
        # Test with extreme values
        c = np.array([1e6])
        G = np.array([[1e6, 1e6]])
        GI = np.array([[1e6]])
        E = np.array([[1, 0], [0, 1]])
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        
        # Should still be finite
        assert np.all(np.isfinite(bounds[0]))
        assert np.all(np.isfinite(bounds[1]))
        assert np.all(bounds[0] <= bounds[1])
        
        # Test with very small values
        c = np.array([1e-6])
        G = np.array([[1e-6, 1e-6]])
        GI = np.array([[1e-6]])
        E = np.array([[1, 0], [0, 1]])
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        
        # Should still be finite
        assert np.all(np.isfinite(bounds[0]))
        assert np.all(np.isfinite(bounds[1]))
        assert np.all(bounds[0] <= bounds[1])
    
    def test_compBoundsPolyZono_integration(self):
        """Test integration with other functions"""
        # Test that bounds can be used for containment checks
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        bounds = compBoundsPolyZono(c, G, GI, E)
        
        # Test that center is within bounds
        assert np.all(bounds[0] <= c)
        assert np.all(c <= bounds[1])
        
        # Test that bounds are reasonable
        # Lower bound should be <= center - sum of generator magnitudes
        expected_lower = c - np.sum(np.abs(G), axis=1) - np.sum(np.abs(GI), axis=1)
        assert np.all(bounds[0] <= expected_lower)
        
        # Upper bound should be >= center + sum of generator magnitudes
        expected_upper = c + np.sum(np.abs(G), axis=1) + np.sum(np.abs(GI), axis=1)
        assert np.all(bounds[1] >= expected_upper)
