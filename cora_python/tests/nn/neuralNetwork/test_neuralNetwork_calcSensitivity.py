"""
Test for neuralNetwork calcSensitivity method

This test verifies that the calcSensitivity method works correctly with different inputs.
"""

import pytest
import numpy as np

def test_neuralNetwork_calcSensitivity_basic():
    """Test calcSensitivity method with basic network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    from cora_python.nn.layers.nonlinear.nnSoftmaxLayer import nnSoftmaxLayer
    
    # Specify number of input and output dimensions
    n0 = 5
    nK = 7
    
    # Generate a random neural network
    np.random.seed(42)
    W1 = np.random.rand(10, n0) * 2 - 1
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(nK, 10) * 2 - 1
    b2 = np.random.rand(nK, 1)
    
    nn = NeuralNetwork([
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2),
        nnSoftmaxLayer()
    ])
    
    # Specify a batch size
    bSz = 13
    
    # Generate a random input
    x = np.random.rand(n0, bSz)
    
    # Compute the output
    y = nn.evaluate(x)
    
    # Calculate the sensitivity
    S = nn.calcSensitivity(x, {}, True)
    
    # Check the dimensions of the sensitivity matrix
    assert S.shape == (nK, n0, bSz)
    
    # Generate a second random input
    x_ = np.random.rand(n0, bSz)
    
    # Compute the difference between the two inputs
    dx = x - x_
    
    # Compute the new output
    y_ = nn.evaluate(x + dx)
    
    # Compute expected difference based on the sensitivity
    dy = np.einsum('ijk,ik->jk', S, dx)
    
    # Check if the directions of the sensitivity matrix are correct
    # (This is a basic check - in practice, the sensitivity should be more accurate)
    assert np.all(np.sign(y + dy) == np.sign(y_))
    
    # Extract the sensitivity matrix of the last layer
    Sk = nn.layers[-2].sensitivity
    
    # Check the dimensions of the sensitivity matrix
    assert Sk.shape == (nK, nK, bSz)

def test_neuralNetwork_calcSensitivity_single_input():
    """Test calcSensitivity method with single input"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    nn = NeuralNetwork([
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ])
    
    # Single input
    x = np.array([[1], [2]])
    
    # Calculate sensitivity
    S = nn.calcSensitivity(x, {}, False)
    
    # Check dimensions
    assert S.shape == (2, 2, 1)

def test_neuralNetwork_calcSensitivity_with_options():
    """Test calcSensitivity method with different options"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    nn = NeuralNetwork([
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ])
    
    # Input
    x = np.array([[1], [2]])
    
    # Options
    options = {
        'nn': {
            'train': {
                'backprop': True
            }
        }
    }
    
    # Calculate sensitivity
    S = nn.calcSensitivity(x, options, True)
    
    # Check dimensions
    assert S.shape == (2, 2, 1)

def test_neuralNetwork_calcSensitivity_no_backprop():
    """Test calcSensitivity method without backprop"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    nn = NeuralNetwork([
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ])
    
    # Input
    x = np.array([[1], [2]])
    
    # Calculate sensitivity without backprop
    S = nn.calcSensitivity(x, {}, False)
    
    # Check dimensions
    assert S.shape == (2, 2, 1)

def test_neuralNetwork_calcSensitivity_empty_network():
    """Test calcSensitivity method with empty network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Create empty network
    nn = NeuralNetwork([])
    
    # Input
    x = np.array([[1], [2]])
    
    # Calculate sensitivity
    S = nn.calcSensitivity(x, {}, False)
    
    # Should return identity matrix for empty network
    assert S.shape == (2, 2, 1)
    assert np.allclose(S[:, :, 0], np.eye(2))
