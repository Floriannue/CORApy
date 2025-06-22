"""
Test file for ellipsoid rank method.
"""

import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


class TestEllipsoidRank:
    """Test class for ellipsoid rank method."""
    
    def test_rank_empty_ellipsoid(self):
        """Test rank of empty ellipsoid."""
        E = Ellipsoid.empty(2)
        assert E.rank() == 0
    
    def test_rank_full_dimensional(self):
        """Test rank of full-dimensional ellipsoid."""
        # Full-dimensional ellipsoid
        Q = np.array([[5.4387811500952807, 12.4977183618314545], 
                      [12.4977183618314545, 29.6662117284481646]])
        q = np.array([[-0.7445068341257537], [3.5800647524843665]])
        E = Ellipsoid(Q, q)
        
        n = E.dim()
        assert E.rank() == n
    
    def test_rank_degenerate_ellipsoid(self):
        """Test rank of degenerate ellipsoid."""
        # Degenerate ellipsoid
        Q = np.array([[4.2533342807136076, 0.6346400221575308], 
                      [0.6346400221575309, 0.0946946398147988]])
        q = np.array([[-2.4653656883489115], [0.2717868749873985]])
        E = Ellipsoid(Q, q)
        
        n = E.dim()
        assert E.rank() != n
        assert E.rank() > 0  # Should still have some rank
    
    def test_rank_zero_matrix(self):
        """Test rank of ellipsoid with zero shape matrix."""
        # Zero shape matrix ellipsoid
        Q = np.array([[0.0, 0.0], [0.0, 0.0]])
        q = np.array([[1.0986933635979599], [-1.9884387759871638]])
        E = Ellipsoid(Q, q)
        
        assert E.rank() == 0
    
    def test_rank_1d_ellipsoid(self):
        """Test rank of 1D ellipsoid."""
        Q = np.array([[2.5]])
        q = np.array([[1.0]])
        E = Ellipsoid(Q, q)
        
        assert E.rank() == 1
    
    def test_rank_1d_degenerate(self):
        """Test rank of 1D degenerate ellipsoid."""
        Q = np.array([[0.0]])
        q = np.array([[1.0]])
        E = Ellipsoid(Q, q)
        
        assert E.rank() == 0
    
    def test_rank_3d_full_dimensional(self):
        """Test rank of 3D full-dimensional ellipsoid."""
        Q = np.eye(3) * 2
        q = np.array([[1.0], [2.0], [3.0]])
        E = Ellipsoid(Q, q)
        
        assert E.rank() == 3
    
    def test_rank_3d_degenerate(self):
        """Test rank of 3D degenerate ellipsoid."""
        # Create a 3D ellipsoid that's degenerate (rank 2)
        Q = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0]])
        q = np.array([[0.0], [0.0], [0.0]])
        E = Ellipsoid(Q, q)
        
        assert E.rank() == 2
    
    def test_rank_consistency_with_dimension(self):
        """Test that rank is consistent with dimension for full-dimensional ellipsoids."""
        # Create several full-dimensional ellipsoids
        for n in [1, 2, 3, 4, 5]:
            Q = np.eye(n) + 0.1 * np.random.randn(n, n)
            Q = Q @ Q.T  # Make positive definite
            q = np.random.randn(n, 1)  # Column vector
            E = Ellipsoid(Q, q)
            
            assert E.rank() == n
    
    def test_rank_vs_isFullDim(self):
        """Test that rank and isFullDim are consistent."""
        # Full-dimensional case
        Q = np.array([[2.0, 0.5], [0.5, 1.0]])
        q = np.array([[1.0], [-1.0]])
        E = Ellipsoid(Q, q)
        
        assert E.rank() == E.dim()
        assert E.isFullDim()
        
        # Degenerate case
        Q = np.array([[1.0, 0.0], [0.0, 0.0]])
        q = np.array([[0.0], [0.0]])
        E = Ellipsoid(Q, q)
        
        assert E.rank() < E.dim()
        assert not E.isFullDim() 