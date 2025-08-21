"""
Test for validateNNoptions method

This test verifies that the validateNNoptions method works correctly with different options.
"""

import pytest
import numpy as np

def test_validateNNoptions_empty():
    """Test validateNNoptions with empty options"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options = validateNNoptions(options)
    
    # Should not raise error and return options

def test_validateNNoptions_full():
    """Test validateNNoptions with full options"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options = validateNNoptions(options)
    
    # Should not raise error and return options

def test_validateNNoptions_default():
    """Test validateNNoptions default values"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options = validateNNoptions(options)
    
    # Check default poly_method
    assert options['nn']['poly_method'] == 'regression'

def test_validateNNoptions_bound_approx():
    """Test validateNNoptions bound_approx field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['bound_approx'] = False
    options = validateNNoptions(options)
    
    assert options['nn']['bound_approx'] == False

def test_validateNNoptions_num_generators():
    """Test validateNNoptions num_generators field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['num_generators'] = 1000
    options = validateNNoptions(options)
    
    assert options['nn']['num_generators'] == 1000

def test_validateNNoptions_max_gens_post():
    """Test validateNNoptions max_gens_post field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['max_gens_post'] = 100
    options = validateNNoptions(options)
    
    assert options['nn']['max_gens_post'] == 100

def test_validateNNoptions_add_approx_error_to_GI():
    """Test validateNNoptions add_approx_error_to_GI field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['add_approx_error_to_GI'] = True
    options = validateNNoptions(options)
    
    assert options['nn']['add_approx_error_to_GI'] == True

def test_validateNNoptions_plot_multi_layer_approx_info():
    """Test validateNNoptions plot_multi_layer_approx_info field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['plot_multi_layer_approx_info'] = True
    options = validateNNoptions(options)
    
    assert options['nn']['plot_multi_layer_approx_info'] == True

def test_validateNNoptions_poly_method():
    """Test validateNNoptions poly_method field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['poly_method'] = 'ridgeregression'
    options = validateNNoptions(options)
    
    assert options['nn']['poly_method'] == 'ridgeregression'

def test_validateNNoptions_reuse_bounds():
    """Test validateNNoptions reuse_bounds field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['reuse_bounds'] = True
    options = validateNNoptions(options)
    
    assert options['nn']['reuse_bounds'] == True

def test_validateNNoptions_max_bounds():
    """Test validateNNoptions max_bounds field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['max_bounds'] = 2
    options = validateNNoptions(options)
    
    assert options['nn']['max_bounds'] == 2

def test_validateNNoptions_do_pre_order_reduction():
    """Test validateNNoptions do_pre_order_reduction field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['do_pre_order_reduction'] = False
    options = validateNNoptions(options)
    
    assert options['nn']['do_pre_order_reduction'] == False

def test_validateNNoptions_remove_GI():
    """Test validateNNoptions remove_GI field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['remove_GI'] = False
    options = validateNNoptions(options)
    
    assert options['nn']['remove_GI'] == False

def test_validateNNoptions_force_approx_lin_at():
    """Test validateNNoptions force_approx_lin_at field"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['force_approx_lin_at'] = 2
    options = validateNNoptions(options)
    
    assert options['nn']['force_approx_lin_at'] == 2

def test_validateNNoptions_error_unknown_poly_method():
    """Test validateNNoptions error for unknown poly_method"""
    from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions
    
    options = {}
    options['nn'] = {}
    options['nn']['poly_method'] = 'unknown'
    
    with pytest.raises(ValueError):
        validateNNoptions(options)
