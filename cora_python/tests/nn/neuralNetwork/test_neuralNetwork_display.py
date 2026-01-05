"""
test_nn_neuralNetwork_display - unit test function for neuralNetwork/display

TRANSLATED FROM: cora_matlab/unitTests/nn/neuralNetwork/test_nn_neuralNetwork_display.m

Syntax:
    pytest cora_python/tests/nn/neuralNetwork/test_neuralNetwork_display.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/evaluate

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       02-October-2023 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork.neuralNetwork import NeuralNetwork
import io
import sys


def test_neuralNetwork_display_empty():
    """
    TRANSLATED TEST - Empty neural network display test
    """
    # empty case
    nn = NeuralNetwork()
    
    display_str = nn.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        nn.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout


def test_neuralNetwork_display_single_layer():
    """
    TRANSLATED TEST - Single layer neural network display test
    """
    try:
        from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
        from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
        
        # single layer - linear
        nn = NeuralNetwork([nnLinearLayer(np.array([[2, 1], [0, -4]]))])
        display_str = nn.display_()
        assert isinstance(display_str, str)
        assert len(display_str) > 0
        
        # single layer - sigmoid
        nn = NeuralNetwork([nnSigmoidLayer()])
        display_str = nn.display_()
        assert isinstance(display_str, str)
        assert len(display_str) > 0
        
    except ImportError:
        pytest.skip("Required layer classes not available")


def test_neuralNetwork_display_larger_network():
    """
    TRANSLATED TEST - Larger neural network display test
    """
    try:
        from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
        from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
        from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
        
        # larger network
        nn = NeuralNetwork([
            nnLinearLayer(np.array([[2, 2], [0, -4], [-1, 2]])),
            nnLeakyReLULayer(),
            nnLinearLayer(np.array([[2, 2, 0], [-4, -1, 2]])),
            nnSigmoidLayer()
        ])
        display_str = nn.display_()
        assert isinstance(display_str, str)
        assert len(display_str) > 0
        
    except ImportError:
        pytest.skip("Required layer classes not available")


def test_neuralNetwork_display_cnn():
    """
    TRANSLATED TEST - CNN neural network display test
    """
    try:
        from cora_python.nn.layers.convolutional.nnConv2DLayer import nnConv2DLayer
        
        # cnn
        nn = NeuralNetwork([nnConv2DLayer(2, 2)])
        display_str = nn.display_()
        assert isinstance(display_str, str)
        assert len(display_str) > 0
        
        # set input size
        nn.setInputSize([10, 10, 1])
        display_str = nn.display_()
        assert isinstance(display_str, str)
        assert len(display_str) > 0
        
    except ImportError:
        pytest.skip("Required layer classes not available")


def test_neuralNetwork_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    nn = NeuralNetwork()
    
    display_str = nn.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        nn.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(nn) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

