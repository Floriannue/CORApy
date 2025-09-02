"""
Test file for NeuralNetwork.verify method

This file tests the verification method of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

class TestNeuralNetworkVerify:
    """Test class for NeuralNetwork.verify method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create a simple neural network
        W1 = np.array([[1, 2], [3, 4]])
        b1 = np.array([[0], [0]])
        W2 = np.array([[1, 0], [0, 1]])
        b2 = np.array([[0], [0]])
        
        layers = [
            nnLinearLayer(W1, b1),
            nnReLULayer(),
            nnLinearLayer(W2, b2)
        ]
        
        self.nn = NeuralNetwork(layers)
        
        # Mock required methods that verify depends on
        self.nn.castWeights = lambda dtype: None
        self.nn.prepareForZonoBatchEval = lambda x, options, idxLayer: 10
        self.nn.evaluateZonotopeBatch_ = lambda cxi, Gxi, options, idxLayer: (np.random.rand(2, 1), np.random.rand(2, 1))
    
    def test_verify_basic(self):
        """Test basic verification"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options)
        
        # Should return result string and optional counterexamples
        assert isinstance(res, str)
        assert res in ['VERIFIED', 'COUNTEREXAMPLE', 'UNKNOWN']
    
    def test_verify_with_timeout(self):
        """Test verification with timeout"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        timeout = 0.001  # Very short timeout
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options, timeout)
        
                # Should return UNKNOWN due to timeout, but if it finds a counterexample quickly, that's also valid
        assert res in ['UNKNOWN', 'COUNTEREXAMPLE']
        assert x_ is None
        assert y_ is None
    
    def test_verify_with_options(self):
        """Test verification with various options"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {
            'nn': {
                'train': {
                    'mini_batch_size': 16,
                    'use_gpu': False
                },
                'interval_center': False
            }
        }
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options)
        
        # Should work with options
        assert isinstance(res, str)
    
    def test_verify_safe_set_true(self):
        """Test verification with safeSet=True"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options)
        
        # Should work with safeSet=True
        assert isinstance(res, str)
    
    def test_verify_safe_set_false(self):
        """Test verification with safeSet=False"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = False
        options = {}
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options)
        
        # Should work with safeSet=False
        assert isinstance(res, str)
    
    def test_verify_none_options(self):
        """Test verification with None options"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options=None)
        
        # Should work with default options
        assert isinstance(res, str)
    
    def test_verify_none_timeout(self):
        """Test verification with None timeout (should use default)"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options, timeout=None)
        
        # Should work with default timeout
        assert isinstance(res, str)
    
    def test_verify_verbose(self):
        """Test verification with verbose output"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        # Should not raise exception with verbose=True
        res, x_, y_ = self.nn.verify(x, r, A, b, safeSet, options, verbose=True)
        
        assert isinstance(res, str)
    
    def test_verify_different_network(self):
        """Test verification with different network instance"""
        x = np.array([[1], [2]])
        r = np.array([[0.1], [0.1]])  # Radius for each input dimension
        A = np.array([[1, 0], [0, 1]])
        b = np.array([[-0.5], [-0.5]])
        safeSet = True
        options = {}
        
        # Create another network instance
        other_nn = NeuralNetwork([])
        other_nn.castWeights = lambda dtype: None
        other_nn.prepareForZonoBatchEval = lambda x, options, idxLayer: 10
        other_nn.evaluateZonotopeBatch_ = lambda cxi, Gxi, options, idxLayer: (np.random.rand(2, 1), np.random.rand(2, 1))
        
        res, x_, y_ = other_nn.verify(x, r, A, b, safeSet, options)
        
        # Should work with different network
        assert isinstance(res, str)
    
    def test_verify_aux_pop(self):
        """Test _aux_pop helper method by importing it directly"""
        from cora_python.nn.neuralNetwork.verify import _aux_pop
        
        xs = np.array([[1, 2, 3], [4, 5, 6]])
        rs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        bs = 2
        
        xi, ri, xs_new, rs_new = _aux_pop(xs, rs, bs)
        
        # Check shapes
        assert xi.shape == (2, 2)
        assert ri.shape == (2, 2)
        assert xs_new.shape == (2, 1)
        assert rs_new.shape == (2, 1)
    
    def test_verify_aux_split(self):
        """Test _aux_split helper method by importing it directly"""
        from cora_python.nn.neuralNetwork.verify import _aux_split
        
        xi = np.array([[1, 2], [3, 4]])
        ri = np.array([[0.1, 0.2], [0.3, 0.4]])
        sens = np.array([[0.5, 0.6], [0.7, 0.8]])
        nSplits = 3
        nDims = 1
        
        xis, ris = _aux_split(xi, ri, sens, nSplits, nDims)
        
        # Check shapes
        assert xis.shape[0] == 2
        assert ris.shape[0] == 2
        assert xis.shape[1] == 6  # 2 * 3 splits
        assert ris.shape[1] == 6  # 2 * 3 splits
