"""
Test file for ellipsoid empty method.
"""

import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


class TestEllipsoidEmpty:
    """Test class for ellipsoid empty method."""
    
    def test_empty_1d(self):
        """Test empty ellipsoid in 1D."""
        n = 1
        E = Ellipsoid.empty(n)
        
        assert E.representsa_('emptySet')
        assert E.dim() == 1
        assert not E.isFullDim()
        assert E.rank() == 0
    
    def test_empty_2d(self):
        """Test empty ellipsoid in 2D."""
        n = 2
        E = Ellipsoid.empty(n)
        
        assert E.representsa_('emptySet')
        assert E.dim() == 2
        assert not E.isFullDim()
        assert E.rank() == 0
    
    def test_empty_5d(self):
        """Test empty ellipsoid in 5D."""
        n = 5
        E = Ellipsoid.empty(n)
        
        assert E.representsa_('emptySet')
        assert E.dim() == 5
        assert not E.isFullDim()
        assert E.rank() == 0
    
    def test_empty_high_dimensional(self):
        """Test empty ellipsoid in high dimensions."""
        for n in [10, 20, 50]:
            E = Ellipsoid.empty(n)
            
            assert E.representsa_('emptySet')
            assert E.dim() == n
            assert not E.isFullDim()
            assert E.rank() == 0
    
    def test_empty_properties(self):
        """Test properties of empty ellipsoid."""
        E = Ellipsoid.empty(3)
        
        # Should have empty shape matrix (0,0)
        expected_Q = np.zeros((0, 0))
        np.testing.assert_allclose(E.Q, expected_Q)
        
        # Center should be empty (3,0) to preserve dimension info
        assert E.q.shape == (3, 0)
        assert E.q.size == 0
        
        # Should be empty object
        assert E.isemptyobject()
    
    def test_empty_generators(self):
        """Test generators of empty ellipsoid."""
        E = Ellipsoid.empty(3)
        G = E.generators()
        
        # Should return empty matrix with correct dimensions
        assert G.shape == (3, 0)
        assert G.size == 0
    
    def test_empty_center(self):
        """Test center of empty ellipsoid."""
        E = Ellipsoid.empty(2)
        c = E.center()
        
        # Center should be zeros(2,0) - empty but with correct first dimension
        assert c.shape == (2, 0)
        assert c.size == 0  # Should be empty (like MATLAB's isempty)
        
        # Test the MATLAB assertions: isempty(c) && size(c,1) == n
        # In NumPy: c.size == 0 && c.shape[0] == n
        assert c.size == 0  # equivalent to MATLAB's isempty(c)
        assert c.shape[0] == 2  # equivalent to MATLAB's size(c,1) == n
    
    def test_empty_comparison_with_regular(self):
        """Test comparison between empty and regular ellipsoids."""
        E_empty = Ellipsoid.empty(2)
        E_regular = Ellipsoid(np.eye(2))
        
        # Empty should be empty, regular should not
        assert E_empty.representsa_('emptySet')
        assert not E_regular.representsa_('emptySet')
        
        # Empty should not be full-dimensional, regular should be
        assert not E_empty.isFullDim()
        assert E_regular.isFullDim()
        
        # Empty should have rank 0, regular should have rank 2
        assert E_empty.rank() == 0
        assert E_regular.rank() == 2
    
    def test_empty_invalid_dimension(self):
        """Test empty ellipsoid with invalid dimensions."""
        # Test with zero dimension
        with pytest.raises((ValueError, AssertionError)):
            Ellipsoid.empty(0)
        
        # Test with negative dimension
        with pytest.raises((ValueError, AssertionError)):
            Ellipsoid.empty(-1)
    
    def test_empty_contains(self):
        """Test containment properties of empty ellipsoid."""
        E = Ellipsoid.empty(2)
        
        # Empty ellipsoid should not contain any point
        point = np.array([0.0, 0.0])
        assert not E.contains_(point)
        
        # Empty ellipsoid should not contain other sets
        other_E = Ellipsoid(np.eye(2))
        assert not E.contains_(other_E)
    
    def test_empty_display(self):
        """Test display of empty ellipsoid doesn't crash."""
        E = Ellipsoid.empty(2)
        
        # Just ensure display method can be called without error
        try:
            E.display()
        except Exception as e:
            pytest.fail(f"Display method failed for empty ellipsoid: {e}")
    
    def test_empty_multiple_dimensions(self):
        """Test empty ellipsoids of various dimensions."""
        dimensions = [1, 2, 3, 4, 5, 10, 15]
        
        for n in dimensions:
            E = Ellipsoid.empty(n)
            
            # Basic properties should hold for all dimensions
            assert E.dim() == n
            assert E.rank() == 0
            assert not E.isFullDim()
            assert E.representsa_('emptySet')
            assert E.isemptyobject()
            
            # Shape matrix should be (0,0) for empty ellipsoids
            assert E.Q.shape == (0, 0)
            np.testing.assert_allclose(E.Q, np.zeros((0, 0)))
            
            # Center should be (n,0) to preserve dimension
            assert E.q.shape == (n, 0)
            
            # Generators should be (n,0)
            G = E.generators()
            assert G.shape == (n, 0) 