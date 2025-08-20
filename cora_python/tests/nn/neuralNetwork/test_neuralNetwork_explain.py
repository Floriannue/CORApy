"""
Test file for NeuralNetwork.explain method

This file tests the explanation method of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

class TestNeuralNetworkExplain:
    """Test class for NeuralNetwork.explain method"""
    
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
        
        # Mock required methods that explain depends on
        self.nn.getInputNeuronOrder = lambda method, x, inputSize: list(range(inputSize[0]))
    
    def test_explain_basic(self):
        """Test basic explanation"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon)
        
        # Should return lists
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
        
        # Check lengths
        assert len(featOrder) == 2  # 2 input features
        assert len(timesPerFeat) == 2
    
    def test_explain_with_verbose(self):
        """Test explanation with verbose output"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, verbose=True)
        
        # Should work with verbose=True
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_with_method(self):
        """Test explanation with different method"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, method='custom')
        
        # Should work with custom method
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_with_featOrderMethod(self):
        """Test explanation with different feature ordering method"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, featOrderMethod='random')
        
        # Should work with random feature ordering
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_with_refineMethod(self):
        """Test explanation with different refinement method"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, refineMethod='custom')
        
        # Should work with custom refinement method
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_with_inputSize(self):
        """Test explanation with custom input size"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        inputSize = [2, 1, 1]
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, inputSize=inputSize)
        
        # Should work with custom input size
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_with_refineSteps(self):
        """Test explanation with refinement steps"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        refineSteps = [1, 2, 3]
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, refineSteps=refineSteps)
        
        # Should work with refinement steps
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_with_bucketType(self):
        """Test explanation with bucket type"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        bucketType = 'dynamic'
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, bucketType=bucketType)
        
        # Should work with bucket type
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_with_delta(self):
        """Test explanation with delta parameter"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        delta = 0.2
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, delta=delta)
        
        # Should work with delta
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_with_timeout(self):
        """Test explanation with timeout"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        timeout = 60.0
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, timeout=timeout)
        
        # Should work with timeout
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_none_inputSize(self):
        """Test explanation with None inputSize (should use default)"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, inputSize=None)
        
        # Should work with default input size
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
    
    def test_explain_list_featOrderMethod(self):
        """Test explanation with list feature ordering method"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        featOrderMethod = [1, 0]  # Custom order
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, featOrderMethod=featOrderMethod)
        
        # Should work with list feature ordering
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
        assert featOrder == [1, 0]
    
    def test_explain_numpy_featOrderMethod(self):
        """Test explanation with numpy array feature ordering method"""
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        featOrderMethod = np.array([1, 0])  # Custom order
        
        idxFreedFeats, featOrder, timesPerFeat = self.nn.explain(x, target, epsilon, featOrderMethod=featOrderMethod)
        
        # Should work with numpy array feature ordering
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
        assert featOrder == [1, 0]
    
    def test_explain_empty_network(self):
        """Test explanation with empty network"""
        empty_nn = NeuralNetwork([])
        empty_nn.getInputNeuronOrder = lambda method, x, inputSize: list(range(inputSize[0]))
        
        x = np.array([[1], [2]])
        target = 0
        epsilon = 0.1
        
        idxFreedFeats, featOrder, timesPerFeat = empty_nn.explain(x, target, epsilon)
        
        # Should handle empty network gracefully
        assert isinstance(idxFreedFeats, list)
        assert isinstance(featOrder, list)
        assert isinstance(timesPerFeat, list)
