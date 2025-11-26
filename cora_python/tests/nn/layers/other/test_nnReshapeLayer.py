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
    """Test evaluateZonotopeBatch with index-based reshape - verify shape handling matches MATLAB"""
    idx_out = np.array([[1, 2], [3, 4]])
    layer = nnReshapeLayer(idx_out)
    
    # Test 1: Standard 3D input (n, 1, batch) - index-based reordering
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
    
    expected_G = np.zeros((4, 2, 2))
    expected_G[:, :, 0] = [[0.1, 0.2], [0.5, 0.6], [0.3, 0.4], [0.7, 0.8]]  # Reordered
    expected_G[:, :, 1] = [[0.9, 1.0], [1.3, 1.4], [1.1, 1.2], [1.5, 1.6]]  # Reordered
    
    assert np.allclose(c_out, expected_c)
    assert np.allclose(G_out, expected_G)
    assert c_out.shape == (4, 1, 2)
    assert G_out.shape == (4, 2, 2)
    
    # Test 2: 2D input (n, 1) - MATLAB aux_reshape keeps 2D shape
    c_2d = np.array([[1.0], [2.0], [3.0], [4.0]])
    G_2d = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    
    c_out_2d, G_out_2d = layer.evaluateZonotopeBatch(c_2d, G_2d, {})
    
    # MATLAB aux_reshape: if input is 2D, output is 2D (not normalized to 3D)
    # Expected: c_out = c_2d[[0, 2, 1, 3], :] in 2D shape (4, 1)
    expected_c_2d = np.array([[1.0], [3.0], [2.0], [4.0]])
    expected_G_2d = np.array([[0.1, 0.2], [0.5, 0.6], [0.3, 0.4], [0.7, 0.8]])
    
    assert np.allclose(c_out_2d, expected_c_2d)
    assert np.allclose(G_out_2d, expected_G_2d)
    assert c_out_2d.shape == (4, 1)  # 2D input -> 2D output
    assert G_out_2d.shape == (4, 2)  # 2D input -> 2D output
    
    # Test 3: Different batch sizes - verify shape handling
    c_batch = np.zeros((4, 1, 5))
    for b in range(5):
        c_batch[:, 0, b] = [1.0 + b, 2.0 + b, 3.0 + b, 4.0 + b]
    
    G_batch = np.zeros((4, 2, 5))
    for b in range(5):
        G_batch[:, :, b] = [[0.1 + b, 0.2 + b], [0.3 + b, 0.4 + b], 
                            [0.5 + b, 0.6 + b], [0.7 + b, 0.8 + b]]
    
    c_out_batch, G_out_batch = layer.evaluateZonotopeBatch(c_batch, G_batch, {})
    
    # Verify shape is preserved
    assert c_out_batch.shape == (4, 1, 5)
    assert G_out_batch.shape == (4, 2, 5)
    
    # Verify reordering for each batch
    for b in range(5):
        expected_c_b = np.array([[1.0 + b], [3.0 + b], [2.0 + b], [4.0 + b]])
        assert np.allclose(c_out_batch[:, 0, b], expected_c_b.flatten())


def test_nnReshapeLayer_evaluateZonotopeBatch_flatten():
    """Test evaluateZonotopeBatch with flatten (-1) case - verify shape handling matches MATLAB"""
    idx_out = [-1]
    layer = nnReshapeLayer(idx_out)
    
    # Test 1: Standard 3D input (n, 1, batch) - flatten case should pass through unchanged
    c = np.zeros((4, 1, 2))
    c[:, 0, 0] = [1.0, 2.0, 3.0, 4.0]
    c[:, 0, 1] = [5.0, 6.0, 7.0, 8.0]
    
    G = np.zeros((4, 2, 2))
    G[:, :, 0] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    G[:, :, 1] = [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]
    
    c_out, G_out = layer.evaluateZonotopeBatch(c, G, {})
    
    # Flatten case should pass through unchanged (MATLAB behavior)
    assert np.allclose(c_out, c)
    assert np.allclose(G_out, G)
    assert c_out.shape == c.shape
    assert G_out.shape == G.shape
    
    # Test 2: 2D input (n, 1) - should be normalized to 3D internally
    c_2d = np.array([[1.0], [2.0], [3.0], [4.0]])
    G_2d = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    
    c_out_2d, G_out_2d = layer.evaluateZonotopeBatch(c_2d, G_2d, {})
    
    # Should pass through unchanged (MATLAB: aux_reshape just passes through for -1)
    assert np.allclose(c_out_2d, c_2d)
    assert np.allclose(G_out_2d, G_2d)
    
    # Test 3: Different batch sizes - verify shape preservation
    c_batch = np.zeros((6, 1, 3))
    c_batch[:, 0, 0] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    c_batch[:, 0, 1] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    c_batch[:, 0, 2] = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    
    G_batch = np.zeros((6, 3, 3))
    for b in range(3):
        G_batch[:, :, b] = np.random.rand(6, 3)
    
    c_out_batch, G_out_batch = layer.evaluateZonotopeBatch(c_batch, G_batch, {})
    
    # Flatten case should preserve all dimensions
    assert c_out_batch.shape == c_batch.shape
    assert G_out_batch.shape == G_batch.shape
    assert np.allclose(c_out_batch, c_batch)
    assert np.allclose(G_out_batch, G_batch)


def test_nnReshapeLayer_evaluateZonotopeBatch_shape_handling():
    """Test evaluateZonotopeBatch shape handling matches MATLAB - various input shapes"""
    # Test 1: Single element reshape (edge case)
    idx_out = np.array([[1]])
    layer = nnReshapeLayer(idx_out)
    
    c = np.array([[[1.0]]])  # (1, 1, 1)
    G = np.array([[[0.1]]])  # (1, 1, 1)
    
    c_out, G_out = layer.evaluateZonotopeBatch(c, G, {})
    
    assert c_out.shape == (1, 1, 1)
    assert G_out.shape == (1, 1, 1)
    assert np.allclose(c_out, c)
    assert np.allclose(G_out, G)
    
    # Test 2: Large reshape with different generator counts
    idx_out = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 9 elements
    layer = nnReshapeLayer(idx_out)
    
    c = np.zeros((9, 1, 2))
    c[:, 0, 0] = np.arange(1, 10)
    c[:, 0, 1] = np.arange(10, 19)
    
    G = np.zeros((9, 5, 2))  # 5 generators
    for b in range(2):
        G[:, :, b] = np.random.rand(9, 5)
    
    c_out, G_out = layer.evaluateZonotopeBatch(c, G, {})
    
    # Verify shape preservation
    assert c_out.shape == (9, 1, 2)
    assert G_out.shape == (9, 5, 2)
    
    # Test 3: Verify MATLAB column-major flattening behavior
    # idx_out = [[1, 2], [3, 4]] -> idx_vec = [1, 3, 2, 4] (column-major)
    idx_out = np.array([[1, 2], [3, 4]])
    layer = nnReshapeLayer(idx_out)
    
    # Create input where each feature is distinct
    c = np.zeros((4, 1, 1))
    c[:, 0, 0] = [10.0, 20.0, 30.0, 40.0]  # Features 0, 1, 2, 3
    
    G = np.zeros((4, 1, 1))
    G[:, 0, 0] = [1.0, 2.0, 3.0, 4.0]
    
    c_out, G_out = layer.evaluateZonotopeBatch(c, G, {})
    
    # MATLAB: idx_out(:) = [1, 3, 2, 4] (1-based) -> [0, 2, 1, 3] (0-based)
    # So output should be: [c[0], c[2], c[1], c[3]] = [10.0, 30.0, 20.0, 40.0]
    expected_c = np.array([[[10.0]], [[30.0]], [[20.0]], [[40.0]]])
    expected_G = np.array([[[1.0]], [[3.0]], [[2.0]], [[4.0]]])
    
    assert np.allclose(c_out, expected_c)
    assert np.allclose(G_out, expected_G)


if __name__ == "__main__":
    test_nn_nnReshapeLayer()
    test_nn_nnReshapeLayer_flatten()
    test_nnReshapeLayer_evaluateZonotopeBatch_index_based()
    test_nnReshapeLayer_evaluateZonotopeBatch_flatten()
    test_nnReshapeLayer_evaluateZonotopeBatch_shape_handling()
    print("test_nn_nnReshapeLayer successful")
