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
    # Output size is [num_indices, 1] where num_indices is the number of elements in idx_out
    assert outputSize == [4, 1]  # 4 elements in idx_out
    
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


def test_nnReshapeLayer_evaluateZonotopeBatch_index_based():
    """Test evaluateZonotopeBatch with index-based reshape"""
    idx_out = np.array([[1, 2], [3, 4]])
    layer = nnReshapeLayer(idx_out)
    
    # Create batch: c is (n, 1, batch), G is (n, q, batch)
    # Input has 4 features, reshape to 4 features in different order
    c = np.zeros((4, 1, 2))
    c[:, 0, 0] = [1.0, 2.0, 3.0, 4.0]
    c[:, 0, 1] = [5.0, 6.0, 7.0, 8.0]
    
    G = np.zeros((4, 2, 2))
    G[:, :, 0] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    G[:, :, 1] = [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]
    
    c_out, G_out = layer.evaluateZonotopeBatch(c, G, {})
    
    # MATLAB: idx_vec = idx_out(:) = [1, 3, 2, 4] (column-major, 1-based)
    # Python: idx0 = [0, 2, 1, 3] (0-based)
    # Expected: c_out = c[[0, 2, 1, 3], :, :]
    expected_c = np.zeros((4, 1, 2))
    expected_c[:, 0, 0] = [1.0, 3.0, 2.0, 4.0]  # Reordered
    expected_c[:, 0, 1] = [5.0, 7.0, 6.0, 8.0]  # Reordered
    
    assert np.allclose(c_out, expected_c)
    assert c_out.shape == (4, 1, 2)
    assert G_out.shape == (4, 2, 2)


def test_nnReshapeLayer_evaluateZonotopeBatch_flatten():
    """Test evaluateZonotopeBatch with flatten (-1) case"""
    idx_out = [-1]
    layer = nnReshapeLayer(idx_out)
    
    # For flatten case, should pass through unchanged
    c = np.zeros((4, 1, 2))
    c[:, 0, 0] = [1.0, 2.0, 3.0, 4.0]
    c[:, 0, 1] = [5.0, 6.0, 7.0, 8.0]
    
    G = np.zeros((4, 2, 2))
    G[:, :, 0] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    G[:, :, 1] = [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]
    
    c_out, G_out = layer.evaluateZonotopeBatch(c, G, {})
    
    # Flatten case should pass through unchanged
    assert np.allclose(c_out, c)
    assert np.allclose(G_out, G)
    assert c_out.shape == c.shape
    assert G_out.shape == G.shape


if __name__ == "__main__":
    test_nn_nnReshapeLayer()
    test_nn_nnReshapeLayer_flatten()
    test_nnReshapeLayer_evaluateZonotopeBatch_index_based()
    test_nnReshapeLayer_evaluateZonotopeBatch_flatten()
    print("test_nn_nnReshapeLayer successful")
