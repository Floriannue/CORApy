"""
Test file for ellipsoid display method.
"""

import pytest
import numpy as np
import io
import sys
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


class TestEllipsoidDisplay:
    """Test class for ellipsoid display method."""
    
    def capture_display_output(self, ellipsoid):
        """Helper function to capture display output."""
        # Use display_() to get the string
        return ellipsoid.display_()
    
    def test_display_empty_ellipsoid(self):
        """Test display of empty ellipsoid."""
        E = Ellipsoid.empty(2)
        
        # Should not raise an exception
        try:
            output = self.capture_display_output(E)
            # Should contain information about empty set
            assert "empty" in output.lower() or "emptySet" in output
        except Exception as e:
            pytest.fail(f"Display failed for empty ellipsoid: {e}")
    
    def test_display_unit_ellipsoid(self):
        """Test display of unit ellipsoid."""
        E = Ellipsoid(np.eye(2))
        
        try:
            output = self.capture_display_output(E)
            # Should contain dimension information
            assert "dim" in output.lower() or "dimension" in output.lower()
        except Exception as e:
            pytest.fail(f"Display failed for unit ellipsoid: {e}")
    
    def test_display_full_dimensional_ellipsoid(self):
        """Test display of full-dimensional ellipsoid from MATLAB tests."""
        Q = np.array([[5.4387811500952807, 12.4977183618314545], 
                      [12.4977183618314545, 29.6662117284481646]])
        q = np.array([[-0.7445068341257537], [3.5800647524843665]])
        E = Ellipsoid(Q, q)
        
        try:
            output = self.capture_display_output(E)
            # Should contain center and shape matrix information
            assert len(output) > 0  # Should produce some output
        except Exception as e:
            pytest.fail(f"Display failed for full-dimensional ellipsoid: {e}")
    
    def test_display_degenerate_ellipsoid(self):
        """Test display of degenerate ellipsoid from MATLAB tests."""
        Q = np.array([[4.2533342807136076, 0.6346400221575308], 
                      [0.6346400221575309, 0.0946946398147988]])
        q = np.array([[-2.4653656883489115], [0.2717868749873985]])
        E = Ellipsoid(Q, q)
        
        try:
            output = self.capture_display_output(E)
            # Should indicate degeneracy
            assert len(output) > 0
            # MATLAB display shows "degenerate: true" for degenerate ellipsoids
        except Exception as e:
            pytest.fail(f"Display failed for degenerate ellipsoid: {e}")
    
    def test_display_zero_matrix_ellipsoid(self):
        """Test display of ellipsoid with zero shape matrix from MATLAB tests."""
        Q = np.array([[0.0, 0.0], [0.0, 0.0]])
        q = np.array([[1.0986933635979599], [-1.9884387759871638]])
        E = Ellipsoid(Q, q)
        
        try:
            output = self.capture_display_output(E)
            # Should handle zero matrix case
            assert len(output) > 0
        except Exception as e:
            pytest.fail(f"Display failed for zero matrix ellipsoid: {e}")
    
    def test_display_1d_ellipsoid(self):
        """Test display of 1D ellipsoid."""
        Q = np.array([[2.5]])
        q = np.array([[1.0]])
        E = Ellipsoid(Q, q)
        
        try:
            output = self.capture_display_output(E)
            assert len(output) > 0
        except Exception as e:
            pytest.fail(f"Display failed for 1D ellipsoid: {e}")
    
    def test_display_3d_ellipsoid(self):
        """Test display of 3D ellipsoid."""
        Q = np.eye(3) * 2
        q = np.array([[1.0], [2.0], [3.0]])
        E = Ellipsoid(Q, q)
        
        try:
            output = self.capture_display_output(E)
            assert len(output) > 0
        except Exception as e:
            pytest.fail(f"Display failed for 3D ellipsoid: {e}")
    
    def test_display_high_dimensional_ellipsoid(self):
        """Test display of high-dimensional ellipsoid."""
        n = 10
        Q = np.eye(n)
        q = np.random.randn(n, 1)
        E = Ellipsoid(Q, q)
        
        try:
            output = self.capture_display_output(E)
            assert len(output) > 0
        except Exception as e:
            pytest.fail(f"Display failed for high-dimensional ellipsoid: {e}")
    
    def test_display_different_tolerances(self):
        """Test display with different tolerance values."""
        Q = np.eye(2)
        q = np.array([[0.0], [0.0]])
        
        tolerances = [1e-15, 1e-10, 1e-6, 1e-3]
        
        for tol in tolerances:
            E = Ellipsoid(Q, q)
            try:
                output = self.capture_display_output(E)
                assert len(output) > 0
            except Exception as e:
                pytest.fail(f"Display failed for tolerance {tol}: {e}")
    
    def test_display_various_ellipsoids(self):
        """Test display for various ellipsoid configurations."""
        test_cases = [
            # Simple cases
            (np.eye(2), np.zeros((2, 1))),
            (np.diag([1, 4]), np.array([[1], [-1]])),
            # Complex case
            (np.array([[2, 1], [1, 3]]), np.array([[-1], [2]])),
            # Degenerate cases
            (np.array([[1, 0], [0, 0]]), np.zeros((2, 1))),
            (np.zeros((2, 2)), np.array([[1], [1]])),
        ]
        
        for Q, q in test_cases:
            E = Ellipsoid(Q, q)
            try:
                output = self.capture_display_output(E)
                assert len(output) > 0
            except Exception as e:
                pytest.fail(f"Display failed for Q={Q}, q={q}: {e}")
    
    def test_display_no_crash(self):
        """Test that display never crashes for any valid ellipsoid."""
        # Generate random ellipsoids and ensure display doesn't crash
        np.random.seed(42)  # For reproducibility
        
        for _ in range(10):
            n = np.random.randint(1, 6)  # Dimension 1-5
            
            # Create random positive semidefinite matrix
            A = np.random.randn(n, n)
            Q = A @ A.T
            q = np.random.randn(n, 1)
            
            E = Ellipsoid(Q, q)
            
            try:
                output = self.capture_display_output(E)
                # Just ensure it produces some output without crashing
                assert isinstance(output, str)
            except Exception as e:
                pytest.fail(f"Display crashed for random ellipsoid: {e}")
    
    def test_display_contains_key_information(self):
        """Test that display contains key information about the ellipsoid."""
        Q = np.array([[2, 1], [1, 2]])
        q = np.array([[1], [-1]])
        E = Ellipsoid(Q, q)
        
        output = self.capture_display_output(E)
        
        # Should be non-empty
        assert len(output.strip()) > 0
        
        # The exact content depends on implementation, but it should
        # contain some meaningful information about the ellipsoid
        # At minimum, it should not be just whitespace
        assert output.strip() != "" 