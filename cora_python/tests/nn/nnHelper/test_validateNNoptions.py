"""
Test for nnHelper.validateNNoptions function

This test verifies that the validateNNoptions function works correctly for neural network options validation.
Based on the MATLAB test: test_nn_nnHelper_validateNNoptions.m
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions


class TestValidateNNoptions:
    """Test class for validateNNoptions function"""
    
    def test_validateNNoptions_empty(self):
        """Test with empty options"""
        options = {}
        result = validateNNoptions(options)
        
        # Should return options with defaults set
        assert isinstance(result, dict)
        assert 'nn' in result
        assert isinstance(result['nn'], dict)
    
    def test_validateNNoptions_full(self):
        """Test with full options"""
        options = {}
        result = validateNNoptions(options)
        # Test that it can be called again with the result
        result2 = validateNNoptions(result)
        assert isinstance(result2, dict)
    
    def test_validateNNoptions_default_poly_method(self):
        """Test default poly_method is 'regression'"""
        options = {}
        result = validateNNoptions(options)
        assert result['nn']['poly_method'] == 'regression'
    
    def test_validateNNoptions_bound_approx(self):
        """Test bound_approx field"""
        options = {'nn': {'bound_approx': False}}
        result = validateNNoptions(options)
        assert result['nn']['bound_approx'] == False
    
    def test_validateNNoptions_num_generators(self):
        """Test num_generators field"""
        options = {'nn': {'num_generators': 1000}}
        result = validateNNoptions(options)
        assert result['nn']['num_generators'] == 1000
    
    def test_validateNNoptions_max_gens_post(self):
        """Test max_gens_post field"""
        options = {'nn': {'max_gens_post': 100}}
        result = validateNNoptions(options)
        assert result['nn']['max_gens_post'] == 100
    
    def test_validateNNoptions_add_approx_error_to_GI(self):
        """Test add_approx_error_to_GI field"""
        options = {'nn': {'add_approx_error_to_GI': True}}
        result = validateNNoptions(options)
        assert result['nn']['add_approx_error_to_GI'] == True
    
    def test_validateNNoptions_plot_multi_layer_approx_info(self):
        """Test plot_multi_layer_approx_info field"""
        options = {'nn': {'plot_multi_layer_approx_info': True}}
        result = validateNNoptions(options)
        assert result['nn']['plot_multi_layer_approx_info'] == True
    
    def test_validateNNoptions_poly_method_ridgeregression(self):
        """Test poly_method field with 'ridgeregression'"""
        options = {'nn': {'poly_method': 'ridgeregression'}}
        result = validateNNoptions(options)
        assert result['nn']['poly_method'] == 'ridgeregression'
    
    def test_validateNNoptions_reuse_bounds(self):
        """Test reuse_bounds field"""
        options = {'nn': {'reuse_bounds': True}}
        result = validateNNoptions(options)
        assert result['nn']['reuse_bounds'] == True
    
    def test_validateNNoptions_max_bounds(self):
        """Test max_bounds field"""
        options = {'nn': {'max_bounds': 2}}
        result = validateNNoptions(options)
        assert result['nn']['max_bounds'] == 2
    
    def test_validateNNoptions_do_pre_order_reduction(self):
        """Test do_pre_order_reduction field"""
        options = {'nn': {'do_pre_order_reduction': False}}
        result = validateNNoptions(options)
        assert result['nn']['do_pre_order_reduction'] == False
    
    def test_validateNNoptions_remove_GI(self):
        """Test remove_GI field"""
        options = {'nn': {'remove_GI': False}}
        result = validateNNoptions(options)
        assert result['nn']['remove_GI'] == False
    
    def test_validateNNoptions_force_approx_lin_at(self):
        """Test force_approx_lin_at field"""
        options = {'nn': {'force_approx_lin_at': 2}}
        result = validateNNoptions(options)
        assert result['nn']['force_approx_lin_at'] == 2
    
    def test_validateNNoptions_invalid_poly_method(self):
        """Test error handling for invalid poly_method"""
        options = {'nn': {'poly_method': 'unknown'}}
        with pytest.raises(ValueError, match="Field options.nn.poly_method has invalid value"):
            validateNNoptions(options)
    
    def test_validateNNoptions_with_training_fields(self):
        """Test with training fields enabled"""
        # When training fields are enabled, poly_method must be one of ['bounds', 'singh', 'center']
        # because 'regression' is not supported for training
        options = {'nn': {'poly_method': 'bounds'}}
        result = validateNNoptions(options, set_train_fields=True)
        
        # Should have training fields
        assert 'train' in result['nn']
        assert isinstance(result['nn']['train'], dict)
        
        # Check some training defaults
        train_opts = result['nn']['train']
        assert 'backprop' in train_opts
        assert 'max_epoch' in train_opts
        assert 'mini_batch_size' in train_opts
        assert 'loss' in train_opts
        
        # poly_method should be preserved
        assert result['nn']['poly_method'] == 'bounds'
    
    def test_validateNNoptions_combined_options(self):
        """Test with multiple options set"""
        options = {
            'nn': {
                'bound_approx': False,
                'num_generators': 500,
                'poly_method': 'ridgeregression',
                'reuse_bounds': True,
                'max_bounds': 3
            }
        }
        result = validateNNoptions(options)
        
        # All options should be preserved
        assert result['nn']['bound_approx'] == False
        assert result['nn']['num_generators'] == 500
        assert result['nn']['poly_method'] == 'ridgeregression'
        assert result['nn']['reuse_bounds'] == True
        assert result['nn']['max_bounds'] == 3
    
    def test_validateNNoptions_edge_cases(self):
        """Test edge cases"""
        # Test with None
        with pytest.raises((TypeError, ValueError)):
            validateNNoptions(None)
        
        # Test with non-dict
        with pytest.raises((TypeError, ValueError)):
            validateNNoptions("not_a_dict")
        
        # Test with empty dict
        result = validateNNoptions({})
        assert isinstance(result, dict)
        assert 'nn' in result