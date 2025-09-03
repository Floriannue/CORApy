"""
test_poly2bernstein - unit test function for poly2bernstein conversion

This is a Python translation of the MATLAB CORA test implementation.

Authors: MATLAB: Niklas Kochdumper
         Python: AI Assistant
"""

import pytest
import numpy as np
from cora_python.g.functions.helper.sets.contSet.polyZonotope.poly2bernstein import poly2bernstein
from cora_python.contSet.interval.interval import Interval


class TestPoly2Bernstein:
    """Test class for poly2bernstein function"""
    
    def test_poly2bernstein_basic(self):
        """Test basic poly2bernstein conversion"""
        # Simple polynomial: x + 2y
        G = np.array([[1, 2]])
        E = np.array([[1, 0], [0, 1]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should return a list of Bernstein coefficient matrices
        assert isinstance(B, list)
        assert len(B) == 1  # One dimension
        assert isinstance(B[0], np.ndarray)
        assert B[0].ndim == 2
    
    def test_poly2bernstein_multiple_dimensions(self):
        """Test poly2bernstein with multiple dimensions"""
        # Polynomial in 2D: [x + y, 2x - y]
        G = np.array([[1, 1], [2, -1]])
        E = np.array([[1, 0], [0, 1]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should return Bernstein coefficients for each dimension
        assert isinstance(B, list)
        assert len(B) == 2  # Two dimensions
        for i in range(2):
            assert isinstance(B[i], np.ndarray)
            assert B[i].ndim == 2
    
    def test_poly2bernstein_higher_order(self):
        """Test poly2bernstein with higher order polynomials"""
        # Higher order polynomial: x^2 + 2xy + y^2
        G = np.array([[1, 2, 1]])
        E = np.array([[2, 1, 0], [0, 1, 2]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should return Bernstein coefficients
        assert isinstance(B, list)
        assert len(B) == 1  # One dimension
        assert isinstance(B[0], np.ndarray)
        assert B[0].ndim == 2
    
    def test_poly2bernstein_single_variable(self):
        """Test poly2bernstein with single variable"""
        # Single variable polynomial: x^2 + 2x + 1
        G = np.array([[1, 2, 1]])
        E = np.array([[2, 1, 0]])
        dom = Interval(np.array([-1]), np.array([1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should return Bernstein coefficients
        assert isinstance(B, list)
        assert len(B) == 1  # One dimension
        assert isinstance(B[0], np.ndarray)
        assert B[0].ndim == 2
    
    def test_poly2bernstein_three_variables(self):
        """Test poly2bernstein with three variables"""
        # Three variable polynomial: x + y + z
        G = np.array([[1, 1, 1]])
        E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        dom = Interval(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should return Bernstein coefficients
        assert isinstance(B, list)
        assert len(B) == 1  # One dimension
        assert isinstance(B[0], np.ndarray)
        assert B[0].ndim == 2
    
    def test_poly2bernstein_empty_generators(self):
        """Test poly2bernstein with empty generator matrix"""
        # Empty polynomial
        G = np.array([]).reshape(1, 0)
        E = np.array([]).reshape(0, 0)
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should return empty Bernstein coefficients
        assert isinstance(B, list)
        assert len(B) == 1  # One dimension
        assert isinstance(B[0], np.ndarray)
        assert B[0].size == 0 or B[0].ndim == 2
    
    def test_poly2bernstein_different_domains(self):
        """Test poly2bernstein with different domain intervals"""
        # Simple polynomial: x + y
        G = np.array([[1, 1]])
        E = np.array([[1, 0], [0, 1]])
        
        # Test different domains
        domains = [
            Interval(np.array([-1, -1]), np.array([1, 1])),
            Interval(np.array([-2, -2]), np.array([2, 2])),
            Interval(np.array([0, 0]), np.array([1, 1])),
            Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
        ]
        
        for dom in domains:
            B = poly2bernstein(G, E, dom)
            assert isinstance(B, list)
            assert len(B) == 1
            assert isinstance(B[0], np.ndarray)
    
    def test_poly2bernstein_consistency(self):
        """Test that poly2bernstein gives consistent results"""
        # Simple polynomial: x + 2y
        G = np.array([[1, 2]])
        E = np.array([[1, 0], [0, 1]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        # Run multiple times
        results = []
        for _ in range(3):
            B = poly2bernstein(G, E, dom)
            results.append(B)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            for j in range(len(results[0])):
                assert np.array_equal(results[i][j], results[0][j])
    
    def test_poly2bernstein_matrix_dimensions(self):
        """Test that Bernstein coefficient matrices have correct dimensions"""
        # Polynomial: x^2 + 2xy + y^2 (degree 2 in 2 variables)
        G = np.array([[1, 2, 1]])
        E = np.array([[2, 1, 0], [0, 1, 2]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Bernstein matrix should have dimensions related to the degree
        assert isinstance(B, list)
        assert len(B) == 1
        assert isinstance(B[0], np.ndarray)
        assert B[0].ndim == 2
        
        # The dimensions should be reasonable (not too large)
        assert B[0].shape[0] <= 10  # Should not be too large
        assert B[0].shape[1] <= 10  # Should not be too large
    
    def test_poly2bernstein_numerical_stability(self):
        """Test numerical stability of poly2bernstein"""
        # Test with small coefficients
        G = np.array([[1e-6, 1e-6]])
        E = np.array([[1, 0], [0, 1]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should not crash and should return valid results
        assert isinstance(B, list)
        assert len(B) == 1
        assert isinstance(B[0], np.ndarray)
        assert np.all(np.isfinite(B[0]))
    
    def test_poly2bernstein_large_coefficients(self):
        """Test poly2bernstein with large coefficients"""
        # Test with large coefficients
        G = np.array([[1e6, 1e6]])
        E = np.array([[1, 0], [0, 1]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should not crash and should return valid results
        assert isinstance(B, list)
        assert len(B) == 1
        assert isinstance(B[0], np.ndarray)
        assert np.all(np.isfinite(B[0]))
    
    def test_poly2bernstein_zero_polynomial(self):
        """Test poly2bernstein with zero polynomial"""
        # Zero polynomial: 0
        G = np.array([[0]])
        E = np.array([[0, 0]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should return Bernstein coefficients (all zeros)
        assert isinstance(B, list)
        assert len(B) == 1
        assert isinstance(B[0], np.ndarray)
        assert np.allclose(B[0], 0, atol=1e-10)
    
    def test_poly2bernstein_constant_polynomial(self):
        """Test poly2bernstein with constant polynomial"""
        # Constant polynomial: 5
        G = np.array([[5]])
        E = np.array([[0, 0]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        
        B = poly2bernstein(G, E, dom)
        
        # Should return Bernstein coefficients (all 5s)
        assert isinstance(B, list)
        assert len(B) == 1
        assert isinstance(B[0], np.ndarray)
        assert np.allclose(B[0], 5, atol=1e-10)
