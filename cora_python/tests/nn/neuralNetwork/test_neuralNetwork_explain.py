"""
Test for neuralNetwork explain method

This test verifies that the explain method works correctly with different parameters.
"""

import pytest
import numpy as np

def test_neuralNetwork_explain_standard():
    """Test explain method with standard method"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Construct network like MATLAB test
    W1 = np.array([[-9, -8, -7], [10, -6, 0], [-6, 2, 5], [4, 4, -8], [-5, -8, 2], 
                    [0, 6, 2], [-7, 10, -2], [0, 8, 6], [1, -3, -2], [3, 9, 2]])
    W2 = np.array([[3, 6, -5, 3, -6, 2, 6, 2, -4, 8], 
                    [4, 1, 7, -3, -4, 4, 2, 0, 2, -1], 
                    [-3, 9, 1, 5, 10, 9, 1, 4, -6, -7]])
    
    nn = NeuralNetwork([
        nnLinearLayer(W1),
        nnSigmoidLayer(),
        nnLinearLayer(W2)
    ])
    
    # Construct input
    x = np.array([[1], [2], [3]])
    label = 1
    
    # Compute explanation
    verbose = False
    epsilon = 0.2
    
    # Method: standard
    method = 'standard'
    idxFreedFeatsStandard = nn.explain(x, label, epsilon, InputSize=[3, 1, 1], 
                                       Method=method, Verbose=verbose)
    
    # Check expected output
    assert np.array_equal(idxFreedFeatsStandard, np.array([3, 2]))

def test_neuralNetwork_explain_abstract_refine():
    """Test explain method with abstract+refine method"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Construct network like MATLAB test
    W1 = np.array([[-9, -8, -7], [10, -6, 0], [-6, 2, 5], [4, 4, -8], [-5, -8, 2], 
                    [0, 6, 2], [-7, 10, -2], [0, 8, 6], [1, -3, -2], [3, 9, 2]])
    W2 = np.array([[3, 6, -5, 3, -6, 2, 6, 2, -4, 8], 
                    [4, 1, 7, -3, -4, 4, 2, 0, 2, -1], 
                    [-3, 9, 1, 5, 10, 9, 1, 4, -6, -7]])
    
    nn = NeuralNetwork([
        nnLinearLayer(W1),
        nnSigmoidLayer(),
        nnLinearLayer(W2)
    ])
    
    # Construct input
    x = np.array([[1], [2], [3]])
    label = 1
    
    # Compute explanation
    verbose = False
    epsilon = 0.2
    
    # Method: abstract+refine
    method = 'abstract+refine'
    idxFreedFeatsStandard = nn.explain(x, label, epsilon, InputSize=[3, 1, 1], 
                                       Method=method, Verbose=verbose)
    
    # Check expected output
    assert np.array_equal(idxFreedFeatsStandard, np.array([3, 2]))

def test_neuralNetwork_explain_different_epsilon():
    """Test explain method with different epsilon values"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Construct simple network
    W1 = np.array([[1, 2], [3, 4]])
    W2 = np.array([[1, 0], [0, 1]])
    
    nn = NeuralNetwork([
        nnLinearLayer(W1),
        nnSigmoidLayer(),
        nnLinearLayer(W2)
    ])
    
    # Construct input
    x = np.array([[1], [2]])
    label = 0
    
    # Test with different epsilon values
    for epsilon in [0.1, 0.2, 0.5]:
        result = nn.explain(x, label, epsilon, InputSize=[2, 1, 1], 
                           Method='standard', Verbose=False)
        
        # Should return array of indices
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

def test_neuralNetwork_explain_different_methods():
    """Test explain method with different methods"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Construct simple network
    W1 = np.array([[1, 2], [3, 4]])
    W2 = np.array([[1, 0], [0, 1]])
    
    nn = NeuralNetwork([
        nnLinearLayer(W1),
        nnSigmoidLayer(),
        nnLinearLayer(W2)
    ])
    
    # Construct input
    x = np.array([[1], [2]])
    label = 0
    epsilon = 0.2
    
    # Test with different methods
    methods = ['standard', 'abstract+refine']
    for method in methods:
        result = nn.explain(x, label, epsilon, InputSize=[2, 1, 1], 
                           Method=method, Verbose=False)
        
        # Should return array of indices
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

def test_neuralNetwork_explain_verbose():
    """Test explain method with verbose output"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Construct simple network
    W1 = np.array([[1, 2], [3, 4]])
    W2 = np.array([[1, 0], [0, 1]])
    
    nn = NeuralNetwork([
        nnLinearLayer(W1),
        nnSigmoidLayer(),
        nnLinearLayer(W2)
    ])
    
    # Construct input
    x = np.array([[1], [2]])
    label = 0
    epsilon = 0.2
    
    # Test with verbose output
    result = nn.explain(x, label, epsilon, InputSize=[2, 1, 1], 
                       Method='standard', Verbose=True)
    
    # Should return array of indices
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1

def test_neuralNetwork_explain_empty_network():
    """Test explain method with empty network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Create empty network
    nn = NeuralNetwork([])
    
    # Construct input
    x = np.array([[1], [2]])
    label = 0
    epsilon = 0.2
    
    # Should handle empty network gracefully
    result = nn.explain(x, label, epsilon, InputSize=[2, 1, 1], 
                       Method='standard', Verbose=False)
    
    # Should return array of indices
    assert isinstance(result, np.ndarray)
