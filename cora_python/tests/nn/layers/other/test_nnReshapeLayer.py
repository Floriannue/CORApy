"""
test_nn_nnReshapeLayer - tests constructor of nnReshapeLayer

Syntax:
    res = test_nn_nnReshapeLayer()

Inputs:
    -

Outputs:
    res - boolean 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       17-January-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'cora_python'))

from cora_python.nn.layers.other.nnReshapeLayer import nnReshapeLayer


def test_nn_nnReshapeLayer():
    """Test constructor of nnReshapeLayer"""
    
    # simple example
    idx_out = np.array([[1, 2], [3, 4]])
    layer = nnReshapeLayer(idx_out)
    
    # check evaluate
    
    # check point
    x = np.array([[1], [2], [3], [4]])
    y = layer.evaluateNumeric(x, {})
    
    # The reshape should reorder the input according to idx_out
    # idx_out = [[1, 2], [3, 4]] means idx_vec = [1, 3, 2, 4] (column-major order)
    # So result should be [input(1), input(3), input(2), input(4)] = [1, 3, 2, 4]
    expected = np.array([[1], [3], [2], [4]])
    assert np.allclose(y, expected)
    
    # Test with different input
    x2 = np.array([[5], [6], [7], [8]])
    y2 = layer.evaluateNumeric(x2, {})
    expected2 = np.array([[5], [7], [6], [8]])  # Same reordering: [5, 7, 6, 8]
    assert np.allclose(y2, expected2)
    
    # Test getOutputSize
    inputSize = [2, 2]
    outputSize = layer.getOutputSize(inputSize)
    assert outputSize == [2, 2]  # Should match shape of idx_out
    
    # gather results
    res = True
    return res


def test_nn_nnReshapeLayer_flatten():
    """Test nnReshapeLayer with flatten operation"""
    
    # Test flatten case with -1
    idx_out = [-1]
    layer = nnReshapeLayer(idx_out)
    
    # check point
    x = np.array([[1], [2], [3], [4]])
    y = layer.evaluateNumeric(x, {})
    
    # Should flatten to 1D
    expected = np.array([1, 2, 3, 4])
    assert np.allclose(y, expected)
    
    # gather results
    res = True
    return res


if __name__ == "__main__":
    test_nn_nnReshapeLayer()
    test_nn_nnReshapeLayer_flatten()
    print("test_nn_nnReshapeLayer successful")
