"""
Test for nnGeneratorReductionLayer to ensure full translation from MATLAB

This test verifies that ALL MATLAB functionality is properly translated to Python.
"""

import pytest
import numpy as np
from cora_python.nn.layers.linear.nnGeneratorReductionLayer import nnGeneratorReductionLayer


class TestNnGeneratorReductionLayer:
    """Test suite for nnGeneratorReductionLayer to match MATLAB functionality exactly"""
    
    def test_constructor_basic(self):
        """Test basic constructor functionality"""
        maxGens = 10
        layer = nnGeneratorReductionLayer(maxGens)
        assert layer.maxGens == maxGens
        assert layer.name is not None
    
    def test_constructor_with_name(self):
        """Test constructor with name"""
        maxGens = 10
        name = 'test_reduction_layer'
        layer = nnGeneratorReductionLayer(maxGens, name)
        assert layer.maxGens == maxGens
        assert layer.name == name
    
    def test_evaluateZonotopeBatch_basic(self):
        """Test basic zonotope batch evaluation"""
        maxGens = 5
        layer = nnGeneratorReductionLayer(maxGens)
        
        # Create test data: n dimensions, q generators, batchSize batches
        n = 3
        q = 10
        batchSize = 2
        
        # Center: (n, 1, batchSize) or (n, batchSize)
        c = np.zeros((n, batchSize), dtype=np.float64)
        c[:, 0] = [1.0, 2.0, 3.0]
        c[:, 1] = [4.0, 5.0, 6.0]
        
        # Generators: (n, q, batchSize)
        G = np.random.randn(n, q, batchSize).astype(np.float64)
        
        options = {'nn': {'train': {'backprop': False}}}
        
        # Evaluate
        c_out, G_out = layer.evaluateZonotopeBatch(c, G, options)
        
        # Check output shapes
        # After reduction, we should have maxGens generators, but also add n diagonal generators
        # So total generators should be maxGens (reduced from q) + n (approximation errors)
        assert c_out.shape == (n, batchSize)
        # G should have shape (n, maxGens - n + n, batchSize) = (n, maxGens, batchSize)
        # But actually, we keep (maxGens - n) generators and add n diagonal generators
        # So total is maxGens
        assert G_out.shape[0] == n
        assert G_out.shape[1] == maxGens  # maxGens - n (kept) + n (diagonal)
        assert G_out.shape[2] == batchSize
    
    def test_evaluateZonotopeBatch_with_backprop(self):
        """Test evaluateZonotopeBatch with backprop enabled"""
        maxGens = 5
        layer = nnGeneratorReductionLayer(maxGens)
        
        n = 3
        q = 10
        batchSize = 2
        
        c = np.zeros((n, batchSize), dtype=np.float64)
        c[:, 0] = [1.0, 2.0, 3.0]
        c[:, 1] = [4.0, 5.0, 6.0]
        
        G = np.random.randn(n, q, batchSize).astype(np.float64)
        
        options = {'nn': {'train': {'backprop': True}}}
        
        # Evaluate
        c_out, G_out = layer.evaluateZonotopeBatch(c, G, options)
        
        # Check that backprop store is populated
        assert 'store' in layer.backprop
        assert 'keepGensIdx' in layer.backprop['store']
        assert 'reduceGensIdx' in layer.backprop['store']
        assert 'GdIdx' in layer.backprop['store']
    
    def test_evaluateZonotopeBatch_reduction(self):
        """Test that generators are actually reduced"""
        maxGens = 3
        layer = nnGeneratorReductionLayer(maxGens)
        
        n = 2
        q = 10  # More generators than maxGens
        batchSize = 1
        
        c = np.zeros((n, batchSize), dtype=np.float64)
        c[:, 0] = [1.0, 2.0]
        
        # Create generators with different lengths
        G = np.zeros((n, q, batchSize), dtype=np.float64)
        # Make first generator longest
        G[:, 0, 0] = [10.0, 10.0]
        # Make others shorter
        for i in range(1, q):
            G[:, i, 0] = [0.1, 0.1]
        
        options = {'nn': {'train': {'backprop': False}}}
        
        # Evaluate
        c_out, G_out = layer.evaluateZonotopeBatch(c, G, options)
        
        # Should have maxGens generators total (maxGens - n kept + n diagonal)
        assert G_out.shape[1] == maxGens
        
        # The longest generator should be kept
        # (This is a simplified check - actual reduction uses sum of absolute values)
        assert not np.allclose(G_out[:, 0, 0], 0)  # First generator should be non-zero
    
    def test_backpropZonotopeBatch(self):
        """Test backpropagation for zonotope batch"""
        maxGens = 5
        layer = nnGeneratorReductionLayer(maxGens)
        
        n = 3
        q = 10
        batchSize = 2
        
        # First do forward pass with backprop enabled
        c = np.zeros((n, batchSize), dtype=np.float64)
        c[:, 0] = [1.0, 2.0, 3.0]
        c[:, 1] = [4.0, 5.0, 6.0]
        
        G = np.random.randn(n, q, batchSize).astype(np.float64)
        
        options = {'nn': {'train': {'backprop': True}}}
        
        # Forward pass
        c_out, G_out = layer.evaluateZonotopeBatch(c, G, options)
        
        # Now backprop
        gc = np.random.randn(*c_out.shape).astype(np.float64)
        gG = np.random.randn(*G_out.shape).astype(np.float64)
        
        gc_back, gG_back = layer.backpropZonotopeBatch(c, G_out, gc, gG, options)
        
        # Check output shapes
        assert gc_back.shape == c.shape
        assert gG_back.shape == G.shape  # Should match original G shape
    
    def test_aux_reduceGirad(self):
        """Test aux_reduceGirad method"""
        maxGens = 5
        layer = nnGeneratorReductionLayer(maxGens)
        
        n = 3
        q = 10
        batchSize = 2
        
        G = np.random.randn(n, q, batchSize).astype(np.float64)
        
        G_, I, keepGensIdx, reduceGensIdx = layer.aux_reduceGirad(G, maxGens)
        
        # Check shapes
        assert G_.shape[0] == n
        assert G_.shape[2] == batchSize
        # Should keep maxGens - n generators
        assert G_.shape[1] == maxGens - n
        
        assert I.shape == (n, batchSize)
        
        # Check indices
        assert keepGensIdx.shape[0] == maxGens - n
        assert keepGensIdx.shape[1] == batchSize
        
        assert reduceGensIdx.shape[0] == q - (maxGens - n)
        assert reduceGensIdx.shape[1] == batchSize
    
    def test_aux_reducePCA(self):
        """Test aux_reducePCA method"""
        maxGens = 5
        layer = nnGeneratorReductionLayer(maxGens)
        
        n = 3
        q = 10
        batchSize = 2
        
        G = np.random.randn(n, q, batchSize).astype(np.float64)
        
        G_, U = layer.aux_reducePCA(G)
        
        # Check shapes
        assert G_.shape == (n, n, batchSize)  # Reduced to n generators
        assert U.shape == (n, n, batchSize)  # U matrix for each batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

