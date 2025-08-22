"""
Test for nnHelper.calcSquared functions

This test verifies that the calcSquared functions work correctly for polynomial zonotope operations.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.calcSquared import calcSquared
from cora_python.nn.nnHelper.calcSquaredG import calcSquaredG
from cora_python.nn.nnHelper.calcSquaredE import calcSquaredE
from cora_python.nn.nnHelper.calcSquaredGInd import calcSquaredGInd


class TestCalcSquared:
    """Test class for calcSquared function"""
    
    def test_calcSquared_basic(self):
        """Test basic calcSquared functionality"""
        # Test data from MATLAB test
        c = 1
        G = np.array([[2, 1.5, 1]])
        GI = np.array([]).reshape(1, 0)  # Empty independent generators
        E = np.array([[1, 0, 3], [0, 1, 1]])
        
        # Test with both dependent and independent generators
        c2, G2, GI2 = calcSquared(c, G, GI, E, c, G, GI, E, True)
        
        # Check that results are numpy arrays
        assert isinstance(c2, np.ndarray)
        assert isinstance(G2, np.ndarray)
        assert isinstance(GI2, np.ndarray)
        
        # Check dimensions
        assert c2.shape == (1, 1)
        assert G2.shape[0] == 1  # Same number of dimensions
        assert GI2.shape[0] == 1  # Same number of dimensions
    
    def test_calcSquared_no_independent(self):
        """Test calcSquared with no independent generators"""
        c = 2
        G = np.array([[1, 0.5]])
        GI = np.array([]).reshape(1, 0)
        E = np.array([[1, 0], [0, 1]])
        
        c2, G2, GI2 = calcSquared(c, G, GI, E, c, G, GI, E, True)
        
        # Check results
        assert isinstance(c2, np.ndarray)
        assert isinstance(G2, np.ndarray)
        assert isinstance(GI2, np.ndarray)
    
    def test_calcSquared_with_independent(self):
        """Test calcSquared with independent generators"""
        c = 1
        G = np.array([[1, 0.5]])
        GI = np.array([[0.3]])
        E = np.array([[1, 0], [0, 1]])
        
        c2, G2, GI2 = calcSquared(c, G, GI, E, c, G, GI, E, False)
        
        # Check results
        assert isinstance(c2, np.ndarray)
        assert isinstance(G2, np.ndarray)
        assert isinstance(GI2, np.ndarray)
    
    def test_calcSquared_different_dimensions(self):
        """Test calcSquared with different dimensional inputs"""
        c1 = 1
        G1 = np.array([[1, 0.5]])
        GI1 = np.array([]).reshape(1, 0)
        E1 = np.array([[1, 0], [0, 1]])
        
        c2 = 2
        G2 = np.array([[0.8, 0.3]])
        GI2 = np.array([]).reshape(1, 0)
        E2 = np.array([[1, 0], [0, 1]])
        
        c_result, G_result, GI_result = calcSquared(c1, G1, GI1, E1, c2, G2, GI2, E2, True)
        
        # Check results
        assert isinstance(c_result, np.ndarray)
        assert isinstance(G_result, np.ndarray)
        assert isinstance(GI_result, np.ndarray)
    
    def test_calcSquared_edge_cases(self):
        """Test calcSquared edge cases"""
        # Test with zero center
        c = 0
        G = np.array([[1, 0.5]])
        GI = np.array([]).reshape(1, 0)
        E = np.array([[1, 0], [0, 1]])
        
        c2, G2, GI2 = calcSquared(c, G, GI, E, c, G, GI, E, True)
        
        # Check results
        assert isinstance(c2, np.ndarray)
        assert isinstance(G2, np.ndarray)
        assert isinstance(GI2, np.ndarray)
        
        # Test with empty generators
        c = 1
        G = np.array([]).reshape(1, 0)
        GI = np.array([]).reshape(1, 0)
        E = np.array([]).reshape(2, 0)
        
        c2, G2, GI2 = calcSquared(c, G, GI, E, c, G, GI, E, True)
        
        # Check results
        assert isinstance(c2, np.ndarray)
        assert isinstance(G2, np.ndarray)
        assert isinstance(GI2, np.ndarray)


class TestCalcSquaredG:
    """Test class for calcSquaredG function"""
    
    def test_calcSquaredG_basic(self):
        """Test basic calcSquaredG functionality"""
        G1 = np.array([[1, 0.5], [0.3, 0.8]])
        G2 = np.array([[0.6, 0.2], [0.4, 0.9]])
        
        result = calcSquaredG(G1, G2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)  # Should be G1.T @ G2
    
    def test_calcSquaredG_different_dimensions(self):
        """Test calcSquaredG with different dimensional matrices"""
        G1 = np.array([[1, 0.5, 0.3]])
        G2 = np.array([[0.6, 0.2], [0.4, 0.9], [0.1, 0.7]])
        
        result = calcSquaredG(G1, G2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)  # Should be G1.T @ G2
    
    def test_calcSquaredG_edge_cases(self):
        """Test calcSquaredG edge cases"""
        # Test with empty matrices
        G1 = np.array([]).reshape(0, 0)
        G2 = np.array([]).reshape(0, 0)
        
        result = calcSquaredG(G1, G2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        
        # Test with single element
        G1 = np.array([[5]])
        G2 = np.array([[3]])
        
        result = calcSquaredG(G1, G2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)


class TestCalcSquaredE:
    """Test class for calcSquaredE function"""
    
    def test_calcSquaredE_basic(self):
        """Test basic calcSquaredE functionality"""
        E1 = np.array([[1, 0, 3], [0, 1, 1]])
        E2 = np.array([[1, 0, 2], [0, 1, 0]])
        
        result = calcSquaredE(E1, E2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)  # Should have same number of rows as E1/E2
    
    def test_calcSquaredE_different_dimensions(self):
        """Test calcSquaredE with different dimensional matrices"""
        E1 = np.array([[1, 0], [0, 1], [1, 1]])
        E2 = np.array([[1, 0, 2], [0, 1, 1]])
        
        result = calcSquaredE(E1, E2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)  # Should have same number of rows as E1
    
    def test_calcSquaredE_edge_cases(self):
        """Test calcSquaredE edge cases"""
        # Test with empty matrices
        E1 = np.array([]).reshape(0, 0)
        E2 = np.array([]).reshape(0, 0)
        
        result = calcSquaredE(E1, E2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        
        # Test with single element
        E1 = np.array([[5]])
        E2 = np.array([[3]])
        
        result = calcSquaredE(E1, E2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)


class TestCalcSquaredGInd:
    """Test class for calcSquaredGInd function"""
    
    def test_calcSquaredGInd_basic(self):
        """Test basic calcSquaredGInd functionality"""
        G1 = np.array([[1, 0.5], [0.3, 0.8]])
        G2 = np.array([[0.6, 0.2], [0.4, 0.9]])
        
        result = calcSquaredGInd(G1, G2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        # The result should be indices, so should be integers
        assert np.issubdtype(result.dtype, np.integer)
    
    def test_calcSquaredGInd_different_dimensions(self):
        """Test calcSquaredGInd with different dimensional matrices"""
        G1 = np.array([[1, 0.5, 0.3]])
        G2 = np.array([[0.6, 0.2], [0.4, 0.9], [0.1, 0.7]])
        
        result = calcSquaredGInd(G1, G2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.integer)
    
    def test_calcSquaredGInd_edge_cases(self):
        """Test calcSquaredGInd edge cases"""
        # Test with empty matrices
        G1 = np.array([]).reshape(0, 0)
        G2 = np.array([]).reshape(0, 0)
        
        result = calcSquaredGInd(G1, G2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.integer)
        
        # Test with single element
        G1 = np.array([[5]])
        G2 = np.array([[3]])
        
        result = calcSquaredGInd(G1, G2)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.integer)


class TestCalcSquaredIntegration:
    """Test class for integration between calcSquared functions"""
    
    def test_calcSquared_integration(self):
        """Test integration between calcSquared functions"""
        # Test data from MATLAB test
        c = 1
        G = np.array([[2, 1.5, 1]])
        GI = np.array([]).reshape(1, 0)
        E = np.array([[1, 0, 3], [0, 1, 1]])
        
        # Test calcSquared
        c2, G2, GI2 = calcSquared(c, G, GI, E, c, G, GI, E, True)
        
        # Test calcSquaredE
        E2 = calcSquaredE(E, E, True)
        
        # Test calcSquaredG
        G_squared = calcSquaredG(G, G)
        
        # Test calcSquaredGInd
        G_indices = calcSquaredGInd(G, G)
        
        # All should work together
        assert isinstance(c2, np.ndarray)
        assert isinstance(G2, np.ndarray)
        assert isinstance(GI2, np.ndarray)
        assert isinstance(E2, np.ndarray)
        assert isinstance(G_squared, np.ndarray)
        assert isinstance(G_indices, np.ndarray)
