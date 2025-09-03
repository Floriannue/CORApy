"""
Test for nnHelper.compBoundsPolyZono function

This test verifies that the compBoundsPolyZono function works correctly for computing bounds of polynomial zonotopes.
Based on MATLAB test: test_nn_nnHelper_computeBoundsPolyZono.m
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.compBoundsPolyZono import compBoundsPolyZono


class TestCompBoundsPolyZono:
    """Test class for compBoundsPolyZono function"""
    
    def test_compBoundsPolyZono_basic(self):
        """Test basic compBoundsPolyZono functionality - matches MATLAB test"""
        # Test with simple case - matches MATLAB test exactly
        c = np.array([0])
        G = np.array([[2, 0, 1]])
        GI = np.array([[0.5, 0.1]])
        E = np.array([[1, 0, 3], [0, 1, 1]])
        
        # Calculate indices like in MATLAB test
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        # Test approximate bounds
        l, u = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        # Check that bounds are returned
        assert isinstance(l, np.ndarray)
        assert isinstance(u, np.ndarray)
        
        # Check dimensions
        assert l.shape == (1, 1)
        assert u.shape == (1, 1)
        
        # Check that lower bound is less than upper bound
        assert l[0, 0] <= u[0, 0]
        
        # Test tighter bounds using splitting
        l2, u2 = compBoundsPolyZono(c, G, GI, E, ind, ind_, False)
        
        # Check that tighter bounds are returned
        assert isinstance(l2, np.ndarray)
        assert isinstance(u2, np.ndarray)
        
        # Check dimensions
        assert l2.shape == (1, 1)
        assert u2.shape == (1, 1)
        
        # Check that lower bound is less than upper bound
        assert l2[0, 0] <= u2[0, 0]
    
    def test_compBoundsPolyZono_no_independent_generators(self):
        """Test compBoundsPolyZono with no independent generators"""
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([]).reshape(1, 0)  # Empty independent generators
        E = np.array([[1, 0], [0, 1]])
        
        # Calculate indices
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        bounds = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        # Check result structure
        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        assert np.all(bounds[0] <= bounds[1])
    
    def test_compBoundsPolyZono_different_dimensions(self):
        """Test compBoundsPolyZono with different dimensions"""
        # 1D case
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        bounds_1d = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        # Check 1D result
        assert bounds_1d[0].shape == (1, 1)
        assert bounds_1d[1].shape == (1, 1)
        
        # 2D case
        c = np.array([1.0, 2.0])
        G = np.array([[1.0, 0.5], [0.3, 0.8]])
        GI = np.array([[0.2], [0.1]])
        E = np.array([[1, 0], [0, 1]])
        
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        bounds_2d = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        # Check 2D result - the function returns (n, 1) where n is the dimension
        assert bounds_2d[0].shape == (2, 1)
        assert bounds_2d[1].shape == (2, 1)
    
    def test_compBoundsPolyZono_different_orders(self):
        """Test compBoundsPolyZono with different polynomial orders"""
        c = np.array([0])
        G = np.array([[1, 2, 3]])
        GI = np.array([[0.1]])
        
        # Order 1
        E1 = np.array([[1, 0, 1], [0, 1, 0]])
        ind1 = np.where(np.prod(1 - np.mod(E1, 2), axis=0) == 1)[0]
        ind1_ = np.setdiff1d(np.arange(E1.shape[1]), ind1)
        
        bounds1 = compBoundsPolyZono(c, G, GI, E1, ind1, ind1_, True)
        
        # Order 2
        E2 = np.array([[2, 1, 0], [0, 1, 2]])
        ind2 = np.where(np.prod(1 - np.mod(E2, 2), axis=0) == 1)[0]
        ind2_ = np.setdiff1d(np.arange(E2.shape[1]), ind2)
        
        bounds2 = compBoundsPolyZono(c, G, GI, E2, ind2, ind2_, True)
        
        # Both should return valid bounds
        assert np.all(bounds1[0] <= bounds1[1])
        assert np.all(bounds2[0] <= bounds2[1])
    
    def test_compBoundsPolyZono_edge_cases(self):
        """Test compBoundsPolyZono edge cases"""
        # Zero center
        c = np.array([0])
        G = np.array([[1]])
        GI = np.array([[0.1]])
        E = np.array([[1], [0]])
        
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        bounds = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        # Should still return valid bounds
        assert np.all(bounds[0] <= bounds[1])
    
    def test_compBoundsPolyZono_accuracy(self):
        """Test compBoundsPolyZono accuracy"""
        # Simple case where we can verify bounds
        c = np.array([1.0])
        G = np.array([[0.5]])
        GI = np.array([[0.1]])
        E = np.array([[1], [0]])
        
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        # Test both approximation methods
        l_approx, u_approx = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        l_exact, u_exact = compBoundsPolyZono(c, G, GI, E, ind, ind_, False)
        
        # Exact bounds should be tighter than approximate bounds
        assert l_exact[0, 0] >= l_approx[0, 0]
        assert u_exact[0, 0] <= u_approx[0, 0]
    
    def test_compBoundsPolyZono_consistency(self):
        """Test compBoundsPolyZono consistency"""
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        # Multiple calls should give same results
        bounds1 = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        bounds2 = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        assert np.allclose(bounds1[0], bounds2[0])
        assert np.allclose(bounds1[1], bounds2[1])
    
    def test_compBoundsPolyZono_error_handling(self):
        """Test compBoundsPolyZono error handling"""
        c = np.array([1.0])
        G = np.array([[1.0, 0.5]])
        GI = np.array([[0.2]])
        E = np.array([[1, 0], [0, 1]])
        
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        # This should raise an error due to dimension mismatch
        with pytest.raises((ValueError, IndexError)):
            compBoundsPolyZono(c, G, GI, E[:, :1], ind, ind_, True)
    
    def test_compBoundsPolyZono_numerical_stability(self):
        """Test compBoundsPolyZono numerical stability"""
        # Test with very small values
        c = np.array([1e-10])
        G = np.array([[1e-10, 1e-12]])
        GI = np.array([[1e-15]])
        E = np.array([[1, 0], [0, 1]])
        
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        bounds = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        # Should still return valid bounds
        assert np.all(np.isfinite(bounds[0]))
        assert np.all(np.isfinite(bounds[1]))
        assert np.all(bounds[0] <= bounds[1])
    
    def test_compBoundsPolyZono_integration(self):
        """Test compBoundsPolyZono integration with other functions"""
        # Test that bounds can be used in interval arithmetic
        c = np.array([1.0])
        G = np.array([[0.5]])
        GI = np.array([[0.1]])
        E = np.array([[1], [0]])
        
        ind = np.where(np.prod(1 - np.mod(E, 2), axis=0) == 1)[0]
        ind_ = np.setdiff1d(np.arange(E.shape[1]), ind)
        
        bounds = compBoundsPolyZono(c, G, GI, E, ind, ind_, True)
        
        # Test interval operations
        interval_width = bounds[1] - bounds[0]
        assert np.all(interval_width >= 0)
        
        # Test that bounds contain the center
        assert np.all(bounds[0] <= c.reshape(-1, 1))
        assert np.all(bounds[1] >= c.reshape(-1, 1))