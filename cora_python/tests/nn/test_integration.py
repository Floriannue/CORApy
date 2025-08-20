"""
Integration test for neural network classes

This test verifies that different layer types can work together in a complete neural network.
"""

import pytest
import numpy as np

def test_simple_neural_network():
    """Test creating and evaluating a simple neural network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    # Create layers for a simple 2-layer network: Input -> Linear -> ReLU -> Linear -> Output
    
    # First layer: 3 inputs -> 4 outputs
    W1 = np.array([[1, 0, -1], [0, 1, 1], [-1, 1, 0], [1, -1, 1]])  # 4x3
    b1 = np.array([[0.1], [0.2], [0.3], [0.4]])  # 4x1
    linear1 = nnLinearLayer(W1, b1, "linear1")
    
    # Activation layer
    relu = nnReLULayer("relu1")
    
    # Second layer: 4 inputs -> 2 outputs
    W2 = np.array([[1, -1, 0, 1], [0, 1, -1, 1]])  # 2x4
    b2 = np.array([[-0.1], [0.1]])  # 2x1
    linear2 = nnLinearLayer(W2, b2, "linear2")
    
    # Create neural network
    layers = [linear1, relu, linear2]
    nn = NeuralNetwork(layers)
    
    # Test basic properties
    assert len(nn) == 3
    assert len(nn.layers) == 3
    
    # Test forward evaluation
    x = np.array([[1], [2], [-1]])  # 3x1 input
    options = {}
    
    # Manual computation for verification:
    # Layer 1: W1 @ x + b1 = [[1,0,-1],[0,1,1],[-1,1,0],[1,-1,1]] @ [[1],[2],[-1]] + [[0.1],[0.2],[0.3],[0.4]]
    #        = [[2],[1],[1],[-2]] + [[0.1],[0.2],[0.3],[0.4]] = [[2.1],[1.2],[1.3],[-1.6]]
    # ReLU:   max(0, x) = [[2.1],[1.2],[1.3],[0]]
    # Layer 2: W2 @ relu_out + b2 = [[1,-1,0,1],[0,1,-1,1]] @ [[2.1],[1.2],[1.3],[0]] + [[-0.1],[0.1]]
    #        = [[0.9],[-0.1]] + [[-0.1],[0.1]] = [[0.8],[0]]
    
    result = nn.evaluate(x, options)
    expected = np.array([[0.8], [0.0]])
    
    assert np.allclose(result, expected, atol=1e-10)

def test_neural_network_neuron_computation():
    """Test that neural network correctly computes neuron counts"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    # Create a simple network: 5 -> 3 -> 2
    W1 = np.random.randn(3, 5)
    b1 = np.random.randn(3, 1)
    linear1 = nnLinearLayer(W1, b1)
    
    relu = nnReLULayer()
    
    W2 = np.random.randn(2, 3)
    b2 = np.random.randn(2, 1)
    linear2 = nnLinearLayer(W2, b2)
    
    layers = [linear1, relu, linear2]
    nn = NeuralNetwork(layers)
    
    # Test that neuron computation works
    assert nn.neurons_in == 5  # Input size from first linear layer
    assert nn.neurons_out == 2  # Output size from last linear layer

def test_neural_network_batch_evaluation():
    """Test neural network evaluation with batch inputs"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    # Simple 2x2 -> 2x2 -> 2x2 network
    W1 = np.array([[1, 0], [0, 1]])  # Identity
    b1 = np.array([[1], [-1]])  # Add bias
    linear1 = nnLinearLayer(W1, b1)
    
    relu = nnReLULayer()
    
    W2 = np.array([[1, 1], [1, -1]])  # Sum and difference
    b2 = np.array([[0], [0]])  # No bias
    linear2 = nnLinearLayer(W2, b2)
    
    layers = [linear1, relu, linear2]
    nn = NeuralNetwork(layers)
    
    # Batch input: 3 samples
    x_batch = np.array([[2, 0, -1], [1, 2, 3]])  # 2x3
    options = {}
    
    result = nn.evaluate(x_batch, options)
    
    # Manual computation for each sample:
    # Sample 1: [2,1] -> [3,0] -> ReLU -> [3,0] -> [3,3]
    # Sample 2: [0,2] -> [1,1] -> ReLU -> [1,1] -> [2,0]  
    # Sample 3: [-1,3] -> [0,2] -> ReLU -> [0,2] -> [2,-2]
    expected = np.array([[3, 2, 2], [3, 0, -2]])
    
    assert result.shape == (2, 3)
    assert np.allclose(result, expected, atol=1e-10)

def test_neural_network_layer_types():
    """Test that neural network works with different layer types"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    # Create network with different activation types
    W1 = np.array([[1, -1]])  # 1x2
    b1 = np.array([[0]])  # 1x1
    linear1 = nnLinearLayer(W1, b1)
    
    # Use LeakyReLU instead of ReLU
    leaky_relu = nnLeakyReLULayer(alpha=0.1)
    
    W2 = np.array([[2]])  # 1x1  
    b2 = np.array([[1]])  # 1x1
    linear2 = nnLinearLayer(W2, b2)
    
    layers = [linear1, leaky_relu, linear2]
    nn = NeuralNetwork(layers)
    
    # Test with negative result from first layer
    x = np.array([[0], [1]])  # Should give W1@x + b1 = [0-1] + [0] = [-1]
    options = {}
    
    result = nn.evaluate(x, options)
    
    # Expected: [-1] -> LeakyReLU(alpha=0.1) -> [-0.1] -> 2*[-0.1] + 1 = [0.8]
    expected = np.array([[0.8]])
    
    assert np.allclose(result, expected, atol=1e-10)
