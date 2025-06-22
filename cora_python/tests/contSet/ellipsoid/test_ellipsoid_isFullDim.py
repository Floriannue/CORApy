"""
Test file for ellipsoid isFullDim method.
"""

import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


class TestEllipsoidIsFullDim:
    """Test class for ellipsoid isFullDim method."""
    
    def test_isFullDim_empty_ellipsoid(self):
        """Test isFullDim of empty ellipsoid."""
        E = Ellipsoid.empty(2)
        assert not E.isFullDim()
    
    def test_isFullDim_full_dimensional(self):
        """Test isFullDim of full-dimensional ellipsoid."""
        # Full-dimensional ellipsoid from MATLAB tests
        Q = np.array([[5.4387811500952807, 12.4977183618314545], 
                      [12.4977183618314545, 29.6662117284481646]])
        q = np.array([[-0.7445068341257537], [3.5800647524843665]])
        E = Ellipsoid(Q, q)
        
        assert E.isFullDim()
    
    def test_isFullDim_degenerate_ellipsoid(self):
        """Test isFullDim of degenerate ellipsoid."""
        # Degenerate ellipsoid from MATLAB tests
        Q = np.array([[4.2533342807136076, 0.6346400221575308], 
                      [0.6346400221575309, 0.0946946398147988]])
        q = np.array([[-2.4653656883489115], [0.2717868749873985]])
        E = Ellipsoid(Q, q)
        
        assert not E.isFullDim()
    
    def test_isFullDim_zero_matrix(self):
        """Test isFullDim of ellipsoid with zero shape matrix."""
        # Zero shape matrix ellipsoid from MATLAB tests
        Q = np.array([[0.0, 0.0], [0.0, 0.0]])
        q = np.array([[1.0986933635979599], [-1.9884387759871638]])
        E = Ellipsoid(Q, q)
        
        assert not E.isFullDim()
    
    def test_isFullDim_1d_full_dimensional(self):
        """Test isFullDim of 1D full-dimensional ellipsoid."""
        Q = np.array([[2.5]])
        q = np.array([[1.0]])
        E = Ellipsoid(Q, q)
        
        assert E.isFullDim()
    
    def test_isFullDim_1d_degenerate(self):
        """Test isFullDim of 1D degenerate ellipsoid."""
        Q = np.array([[0.0]])
        q = np.array([[1.0]])
        E = Ellipsoid(Q, q)
        
        assert not E.isFullDim()
    
    def test_isFullDim_3d_full_dimensional(self):
        """Test isFullDim of 3D full-dimensional ellipsoid."""
        Q = np.eye(3) * 2
        q = np.array([[1.0], [2.0], [3.0]])
        E = Ellipsoid(Q, q)
        
        assert E.isFullDim()
    
    def test_isFullDim_3d_degenerate(self):
        """Test isFullDim of 3D degenerate ellipsoid."""
        # Create a 3D ellipsoid that's degenerate (rank 2)
        Q = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0]])
        q = np.array([[0.0], [0.0], [0.0]])
        E = Ellipsoid(Q, q)
        
        assert not E.isFullDim()
    
    def test_isFullDim_identity_matrix(self):
        """Test isFullDim with identity matrix."""
        # Unit ellipsoid - should be full dimensional
        Q = np.eye(2)
        E = Ellipsoid(Q)
        
        assert E.isFullDim()
    
    def test_isFullDim_singular_matrix(self):
        """Test isFullDim with singular matrix."""
        # Singular matrix - should not be full dimensional
        Q = np.array([[1.0, 1.0], [1.0, 1.0]])
        E = Ellipsoid(Q)
        
        assert not E.isFullDim()
    
    def test_isFullDim_consistency_with_rank(self):
        """Test that isFullDim is consistent with rank and dimension."""
        # Test several cases
        test_cases = [
            # Full-dimensional cases
            (np.eye(2), True),
            (np.diag([1, 2]), True),
            (np.array([[2, 1], [1, 2]]), True),
            # Degenerate cases
            (np.zeros((2, 2)), False),
            (np.array([[1, 0], [0, 0]]), False),
            (np.array([[1, 1], [1, 1]]), False),
        ]
        
        for Q, expected_full_dim in test_cases:
            E = Ellipsoid(Q)
            assert E.isFullDim() == expected_full_dim
            assert E.isFullDim() == (E.rank() == E.dim())
    
    def test_isFullDim_high_dimensional(self):
        """Test isFullDim for higher dimensional ellipsoids."""
        # 5D full-dimensional ellipsoid
        Q = np.eye(5) * 2
        E = Ellipsoid(Q)
        assert E.isFullDim()
        
        # 5D degenerate ellipsoid (rank 3)
        Q = np.eye(5)
        Q[3, 3] = 0  # Make one dimension degenerate
        Q[4, 4] = 0  # Make another dimension degenerate
        E = Ellipsoid(Q)
        assert not E.isFullDim()
    
    def test_isFullDim_numerical_precision(self):
        """Test isFullDim with numerical precision considerations."""
        # Create a nearly singular matrix
        Q = np.array([[1.0, 0.0], [0.0, 1e-12]])
        E = Ellipsoid(Q)
        
        # Should be considered degenerate due to small eigenvalue
        assert not E.isFullDim() 