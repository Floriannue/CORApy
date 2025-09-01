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
        assert layer.scale == scale
        assert layer.offset == offset
    
    def test_nnElementwiseAffineLayer_constructor_basic_vector(self):
        """Test basic constructor with vector values - matches MATLAB test exactly"""
        # matches MATLAB: scale = [2;2]; offset = [-4;-4]; layer = nnElementwiseAffineLayer(scale, offset);
        scale = np.array([[2], [2]])  # Column vector like MATLAB
        offset = np.array([[-4], [-4]])  # Column vector like MATLAB
        layer = nnElementwiseAffineLayer(scale, offset)
        
        # matches MATLAB: assert(compareMatrices(layer.scale, scale)); assert(compareMatrices(layer.offset, offset));
        np.testing.assert_array_equal(layer.scale, scale)
        np.testing.assert_array_equal(layer.offset, offset)
    
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
        assert np.sum(layer.offset) == 0
    
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


if __name__ == '__main__':
    pytest.main([__file__])
