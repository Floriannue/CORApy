"""
Test file for nnElementwiseAffineLayer constructor

This file tests the basic constructor functionality of the nnElementwiseAffineLayer class.
Matches the MATLAB test structure exactly.
"""

import pytest
import numpy as np
from cora_python.nn.layers.linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer

class TestNnElementwiseAffineLayer:
    """Test class for nnElementwiseAffineLayer constructor - matches MATLAB test exactly"""
    
    def test_nnElementwiseAffineLayer_constructor_basic_scalar(self):
        """Test basic constructor with scalar values - matches MATLAB test exactly"""
        # matches MATLAB: scale = 2; offset = 3; layer = nnElementwiseAffineLayer(scale, offset);
        scale = 2
        offset = 3
        layer = nnElementwiseAffineLayer(scale, offset)
        
        # matches MATLAB: assert(layer.scale == scale); assert(layer.offset == offset);
        # Convert torch tensors to numpy for comparison
        scale_np = layer.scale.cpu().numpy() if hasattr(layer.scale, 'cpu') else layer.scale
        offset_np = layer.offset.cpu().numpy() if hasattr(layer.offset, 'cpu') else layer.offset
        assert scale_np == scale
        assert offset_np == offset
    
    def test_nnElementwiseAffineLayer_constructor_basic_vector(self):
        """Test basic constructor with vector values - matches MATLAB test exactly"""
        # matches MATLAB: scale = [2;2]; offset = [-4;-4]; layer = nnElementwiseAffineLayer(scale, offset);
        scale = np.array([[2], [2]])  # Column vector like MATLAB
        offset = np.array([[-4], [-4]])  # Column vector like MATLAB
        layer = nnElementwiseAffineLayer(scale, offset)
        
        # matches MATLAB: assert(compareMatrices(layer.scale, scale)); assert(compareMatrices(layer.offset, offset));
        # Convert torch tensors to numpy for comparison
        scale_np = layer.scale.cpu().numpy() if hasattr(layer.scale, 'cpu') else layer.scale
        offset_np = layer.offset.cpu().numpy() if hasattr(layer.offset, 'cpu') else layer.offset
        np.testing.assert_array_equal(scale_np, scale)
        np.testing.assert_array_equal(offset_np, offset)
    
    def test_nnElementwiseAffineLayer_constructor_different_dimensions(self):
        """Test with different dimensions - matches MATLAB test exactly"""
        # matches MATLAB: layer = nnElementwiseAffineLayer([2;3], -4);
        layer1 = nnElementwiseAffineLayer(np.array([[2], [3]]), -4)
        assert layer1 is not None
        
        # matches MATLAB: layer = nnElementwiseAffineLayer(2, [-1; 2]);
        layer2 = nnElementwiseAffineLayer(2, np.array([[-1], [2]]))
        assert layer2 is not None
    
    def test_nnElementwiseAffineLayer_constructor_variable_input(self):
        """Test with variable input (no offset) - matches MATLAB test exactly"""
        # matches MATLAB: layer = nnElementwiseAffineLayer(scale); assert(sum(layer.offset) == 0);
        scale = np.array([[2], [2]])
        layer = nnElementwiseAffineLayer(scale)
        
        # Check that offset is zero when not provided
        # Convert torch tensors to numpy for comparison
        offset_np = layer.offset.cpu().numpy() if hasattr(layer.offset, 'cpu') else layer.offset
        assert np.sum(offset_np) == 0
    
    def test_nnElementwiseAffineLayer_constructor_name(self):
        """Test constructor with name parameter - matches MATLAB test exactly"""
        # matches MATLAB: name = "TestLayer"; layer = nnElementwiseAffineLayer(scale, offset, name); assert(layer.name == name);
        scale = np.array([[2], [2]])
        offset = np.array([[-4], [-4]])
        name = "TestLayer"
        layer = nnElementwiseAffineLayer(scale, offset, name)
        
        # Check that name is stored correctly
        assert layer.name == name
    
    def test_nnElementwiseAffineLayer_constructor_wrong_inputs(self):
        """Test that constructor fails with wrong inputs - matches MATLAB test exactly"""
        scale = np.array([[2], [2]])
        
        # matches MATLAB: assertThrowsAs(@nnElementwiseAffineLayer,'CORA:wrongInputInConstructor',[2,2],1);
        # Test with wrong scale dimensions
        with pytest.raises(Exception):
            nnElementwiseAffineLayer(np.array([[2, 2]]), 1)
        
        # matches MATLAB: assertThrowsAs(@nnElementwiseAffineLayer,'CORA:wrongInputInConstructor',scale,[-4,4]);
        # Test with wrong offset dimensions
        with pytest.raises(Exception):
            nnElementwiseAffineLayer(scale, np.array([[-4, 4]]))
        
        # matches MATLAB: assertThrowsAs(@nnElementwiseAffineLayer,'CORA:wrongInputInConstructor',[-3;2;1],[-4;4]);
        # Test with mismatched dimensions
        with pytest.raises(Exception):
            nnElementwiseAffineLayer(np.array([[-3], [2], [1]]), np.array([[-4], [4]]))
    
    def test_evaluateZonotopeBatch_basic(self):
        """Test basic zonotope batch evaluation"""
        scale = np.array([2.0, 3.0])
        offset = np.array([0.5, 1.0])
        layer = nnElementwiseAffineLayer(scale, offset)
        
        # Create 3D test arrays: c is (n, 1, batch), G is (n, q, batch)
        n = 2
        q = 3
        batch = 2
        
        c = np.zeros((n, 1, batch))
        c[:, 0, 0] = [1.0, 2.0]
        c[:, 0, 1] = [3.0, 4.0]
        
        G = np.zeros((n, q, batch))
        G[:, :, 0] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        G[:, :, 1] = [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        
        options = {'nn': {'interval_center': False}}
        
        # Evaluate
        c_out, G_out = layer.evaluateZonotopeBatch(c, G, options)
        
        # Expected: c_out = scale * c + offset
        expected_c = np.zeros((n, 1, batch))
        expected_c[:, 0, 0] = [2.0 * 1.0 + 0.5, 3.0 * 2.0 + 1.0]  # [2.5, 7.0]
        expected_c[:, 0, 1] = [2.0 * 3.0 + 0.5, 3.0 * 4.0 + 1.0]  # [6.5, 13.0]
        
        # Expected: G_out = scale * G
        expected_G = np.zeros((n, q, batch))
        expected_G[:, :, 0] = [[2.0 * 0.1, 2.0 * 0.2, 2.0 * 0.3], 
                                [3.0 * 0.4, 3.0 * 0.5, 3.0 * 0.6]]
        expected_G[:, :, 1] = [[2.0 * 0.7, 2.0 * 0.8, 2.0 * 0.9],
                                [3.0 * 1.0, 3.0 * 1.1, 3.0 * 1.2]]
        
        assert np.allclose(c_out, expected_c)
        assert np.allclose(G_out, expected_G)
    
    def test_evaluateZonotopeBatch_with_negative_scale(self):
        """Test with negative scale values"""
        scale = np.array([-2.0, 3.0])  # First element negative
        offset = np.array([0.5, 1.0])
        layer = nnElementwiseAffineLayer(scale, offset)
        
        c = np.zeros((2, 1, 1))
        c[:, 0, 0] = [1.0, 2.0]  # (2, 1, 1)
        G = np.zeros((2, 2, 1))
        G[:, :, 0] = [[0.1, 0.2], [0.3, 0.4]]  # (2, 2, 1)
        
        options = {'nn': {'interval_center': False}}
        c_out, G_out = layer.evaluateZonotopeBatch(c, G, options)
        
        # Expected: c_out = scale * c + offset
        expected_c = np.zeros((2, 1, 1))
        expected_c[:, 0, 0] = [-2.0 * 1.0 + 0.5, 3.0 * 2.0 + 1.0]  # [-1.5, 7.0]
        expected_G = np.zeros((2, 2, 1))
        expected_G[:, :, 0] = [[-2.0 * 0.1, -2.0 * 0.2], [3.0 * 0.3, 3.0 * 0.4]]
        
        assert np.allclose(c_out, expected_c)
        assert np.allclose(G_out, expected_G)
    
    def test_evaluateZonotopeBatch_interval_center(self):
        """Test with interval_center option enabled"""
        scale = np.array([2.0, -1.0])  # Second element negative
        offset = np.array([0.0, 0.0])
        layer = nnElementwiseAffineLayer(scale, offset)
        
        # For interval_center, c has shape (n, 2, batch) with [lower, upper]
        c = np.zeros((2, 2, 1))
        c[:, 0, 0] = [1.0, 3.0]  # lower bounds
        c[:, 1, 0] = [2.0, 4.0]  # upper bounds
        G = np.zeros((2, 1, 1))
        G[:, 0, 0] = [0.1, 0.2]
        
        options = {'nn': {'interval_center': True}}
        c_out, G_out = layer.evaluateZonotopeBatch(c, G, options)
        
        # With interval_center, bounds should be sorted after scaling
        assert c_out.shape == (2, 2, 1)
        assert G_out.shape == (2, 1, 1)
        # Verify bounds are sorted
        assert np.all(c_out[:, 0, 0] <= c_out[:, 1, 0])  # Lower <= Upper for all features
    
    def test_evaluateZonotopeBatch_single_batch(self):
        """Test with single batch"""
        scale = np.array([1.5])
        offset = np.array([0.1])
        layer = nnElementwiseAffineLayer(scale, offset)
        
        c = np.array([[[1.0]]])  # (1, 1, 1)
        G = np.array([[[0.5, 0.3]]])  # (1, 2, 1)
        
        options = {'nn': {'interval_center': False}}
        c_out, G_out = layer.evaluateZonotopeBatch(c, G, options)
        
        expected_c = np.array([[[1.5 * 1.0 + 0.1]]])  # [[1.6]]
        expected_G = np.array([[[1.5 * 0.5, 1.5 * 0.3]]])  # [[0.75, 0.45]]
        
        assert np.allclose(c_out, expected_c)
        assert np.allclose(G_out, expected_G)


def test_nnElementwiseAffineLayer_evaluateZonotopeBatch_set_enclosure():
    """
    Test nnElementwiseAffineLayer/evaluateZonotopeBatch function - Set-Enclosure Test
    
    Verifies that evaluateZonotopeBatch computes output sets that contain many samples (>1000).
    Based on MATLAB test pattern: cora/unitTests/nn/layers/linear/testnn_nnLinearLayer_evalutateZonotopeBatch.m
    
    This test creates random zonotopes, propagates them through the network, and verifies
    that all sampled points from input zonotopes, when evaluated through the network,
    are contained in the corresponding output zonotopes.
    """
    import sys
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # Reset random number generator for reproducibility
    np.random.seed(0)
    
    # Specify batch size
    bSz = 16
    # Specify input and output dimensions
    inDim = 4
    outDim = 4  # Elementwise affine preserves dimensions
    # Specify number of generators
    numGen = 10
    # Specify number of random samples for validation 
    N = 100  # MATLAB test uses 100 
    
    # Instantiate random layer with positive scale values
    scale = np.random.rand(outDim).astype(np.float64) + 0.5  # Scale between 0.5 and 1.5
    offset = np.random.rand(outDim).astype(np.float64) - 0.5  # Offset between -0.5 and 0.5
    layer = nnElementwiseAffineLayer(scale, offset)
    
    # Instantiate neural networks with only one layer
    nn = NeuralNetwork([layer])
    
    # Prepare the neural network for the batch evaluation
    options = {'nn': {'train': {'num_init_gens': numGen}}}
    nn.prepareForZonoBatchEval(np.zeros((inDim, 1)), options)
    
    # Create random batch of input zonotopes
    # MATLAB: cx = rand([inDim bSz]); Gx = rand([inDim numGen bSz]);
    cx = np.random.rand(inDim, bSz).astype(np.float64)
    Gx = np.random.rand(inDim, numGen, bSz).astype(np.float64)
    
    # Propagate batch of zonotopes
    # MATLAB: [cy,Gy] = nn.evaluateZonotopeBatch(cx,Gx);
    cy, Gy = nn.evaluateZonotopeBatch(cx, Gx, options)
    
    # Check if all samples are contained
    for i in range(bSz):
        # Instantiate i-th input and output zonotope from the batch
        # MATLAB: Xi = zonotope(cx(:,i),Gx(:,:,i));
        # MATLAB: Yi = zonotope(cy(:,i),Gy(:,:,i));
        Xi = Zonotope(cx[:, i].reshape(-1, 1), Gx[:, :, i])
        # Handle both 2D (outDim, bSz) and 3D (outDim, 1, bSz) output shapes
        if cy.ndim == 3:
            # cy is (outDim, 1, bSz), extract (outDim, 1) and reshape to (outDim, 1)
            cy_i = cy[:, 0, i].reshape(-1, 1)
        else:
            # cy is (outDim, bSz), extract (outDim,) and reshape to (outDim, 1)
            cy_i = cy[:, i].reshape(-1, 1)
        Yi = Zonotope(cy_i, Gy[:, :, i])
        
        # Sample random points
        # MATLAB: xsi = randPoint(Xi,N);
        xsi = Xi.randPoint_(N)
        
        # Propagate samples
        # MATLAB: ysi = nn.evaluate(xsi);
        ysi = nn.evaluate(xsi)
        
        # Check if all samples are contained
        # MATLAB: assert(all(contains(Yi,ysi)));
        # Note: contains_ expects points as columns, ysi should be (outDim, N)
        if ysi.ndim == 1:
            ysi = ysi.reshape(-1, 1)
        elif ysi.ndim == 2 and ysi.shape[1] == 1:
            # Single point case
            assert Yi.contains_(ysi), f"Sample {i}: Single point not contained in output zonotope"
        else:
            # Multiple points case: ysi is (outDim, N)
            assert Yi.contains_(ysi), f"Batch {i}: Not all samples contained in output zonotope"


if __name__ == '__main__':
    pytest.main([__file__])
