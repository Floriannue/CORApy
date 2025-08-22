"""
Test for nnHelper.validateNNoptions function

This test verifies that the validateNNoptions function works correctly for neural network options validation.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions


class TestValidateNNoptions:
    """Test class for validateNNoptions function"""
    
    def test_validateNNoptions_basic(self):
        """Test basic validateNNoptions functionality"""
        # Test with empty options
        options = {}
        result = validateNNoptions(options)
        
        # Should return options with defaults set
        assert isinstance(result, dict)
        assert 'method' in result
        assert 'order' in result
        assert 'taylorTerms' in result
        assert 'maxError' in result
        assert 'verbose' in result
    
    def test_validateNNoptions_existing_values(self):
        """Test that existing values are preserved"""
        options = {
            'method': 'exact',
            'order': 5,
            'taylorTerms': 10
        }
        
        result = validateNNoptions(options)
        
        # Existing values should be preserved
        assert result['method'] == 'exact'
        assert result['order'] == 5
        assert result['taylorTerms'] == 10
        
        # Defaults should be set for missing fields
        assert 'maxError' in result
        assert 'verbose' in result
    
    def test_validateNNoptions_method_validation(self):
        """Test method field validation"""
        # Test valid methods
        valid_methods = ['exact', 'approx', 'sampled']
        
        for method in valid_methods:
            options = {'method': method}
            result = validateNNoptions(options)
            assert result['method'] == method
        
        # Test invalid method
        with pytest.raises(ValueError):
            validateNNoptions({'method': 'invalid_method'})
    
    def test_validateNNoptions_order_validation(self):
        """Test order field validation"""
        # Test valid orders
        valid_orders = [1, 2, 3, 5, 10]
        
        for order in valid_orders:
            options = {'order': order}
            result = validateNNoptions(options)
            assert result['order'] == order
        
        # Test invalid orders
        invalid_orders = [0, -1, 1.5, 'invalid']
        
        for order in invalid_orders:
            with pytest.raises(ValueError):
                validateNNoptions({'order': order})
    
    def test_validateNNoptions_taylorTerms_validation(self):
        """Test taylorTerms field validation"""
        # Test valid taylorTerms
        valid_terms = [1, 2, 3, 5, 10]
        
        for terms in valid_terms:
            options = {'taylorTerms': terms}
            result = validateNNoptions(options)
            assert result['taylorTerms'] == terms
        
        # Test invalid taylorTerms
        invalid_terms = [0, -1, 1.5, 'invalid']
        
        for terms in invalid_terms:
            with pytest.raises(ValueError):
                validateNNoptions({'taylorTerms': terms})
    
    def test_validateNNoptions_maxError_validation(self):
        """Test maxError field validation"""
        # Test valid maxError values
        valid_errors = [0.1, 0.01, 0.001, 1e-6]
        
        for error in valid_errors:
            options = {'maxError': error}
            result = validateNNoptions(options)
            assert result['maxError'] == error
        
        # Test invalid maxError values
        invalid_errors = [-0.1, 0, 'invalid', None]
        
        for error in invalid_errors:
            with pytest.raises(ValueError):
                validateNNoptions({'maxError': error})
    
    def test_validateNNoptions_verbose_validation(self):
        """Test verbose field validation"""
        # Test valid verbose values
        valid_verbose = [True, False, 0, 1]
        
        for verbose in valid_verbose:
            options = {'verbose': verbose}
            result = validateNNoptions(options)
            assert result['verbose'] == bool(verbose)
        
        # Test invalid verbose values
        invalid_verbose = ['true', 'false', 2, -1]
        
        for verbose in invalid_verbose:
            with pytest.raises(ValueError):
                validateNNoptions({'verbose': verbose})
    
    def test_validateNNoptions_training_fields(self):
        """Test training-related fields"""
        options = {}
        result = validateNNoptions(options)
        
        # Training fields should be set with defaults
        assert 'train' in result
        assert isinstance(result['train'], dict)
        
        train_opts = result['train']
        assert 'method' in train_opts
        assert 'epochs' in train_opts
        assert 'batchSize' in train_opts
        assert 'learningRate' in train_opts
    
    def test_validateNNoptions_training_method_validation(self):
        """Test training method validation"""
        # Test valid training methods
        valid_methods = ['sgd', 'adam', 'rmsprop']
        
        for method in valid_methods:
            options = {'train': {'method': method}}
            result = validateNNoptions(options)
            assert result['train']['method'] == method
        
        # Test invalid training method
        with pytest.raises(ValueError):
            validateNNoptions({'train': {'method': 'invalid'}})
    
    def test_validateNNoptions_epochs_validation(self):
        """Test epochs validation"""
        # Test valid epochs
        valid_epochs = [1, 10, 100, 1000]
        
        for epochs in valid_epochs:
            options = {'train': {'epochs': epochs}}
            result = validateNNoptions(options)
            assert result['train']['epochs'] == epochs
        
        # Test invalid epochs
        invalid_epochs = [0, -1, 1.5, 'invalid']
        
        for epochs in invalid_epochs:
            with pytest.raises(ValueError):
                validateNNoptions({'train': {'epochs': epochs}})
    
    def test_validateNNoptions_batchSize_validation(self):
        """Test batchSize validation"""
        # Test valid batch sizes
        valid_sizes = [1, 16, 32, 64, 128]
        
        for size in valid_sizes:
            options = {'train': {'batchSize': size}}
            result = validateNNoptions(options)
            assert result['train']['batchSize'] == size
        
        # Test invalid batch sizes
        invalid_sizes = [0, -1, 1.5, 'invalid']
        
        for size in invalid_sizes:
            with pytest.raises(ValueError):
                validateNNoptions({'train': {'batchSize': size}})
    
    def test_validateNNoptions_learningRate_validation(self):
        """Test learningRate validation"""
        # Test valid learning rates
        valid_rates = [0.1, 0.01, 0.001, 1e-4]
        
        for rate in valid_rates:
            options = {'train': {'learningRate': rate}}
            result = validateNNoptions(options)
            assert result['train']['learningRate'] == rate
        
        # Test invalid learning rates
        invalid_rates = [-0.1, 0, 1.1, 'invalid']
        
        for rate in invalid_rates:
            with pytest.raises(ValueError):
                validateNNoptions({'train': {'learningRate': rate}})
    
    def test_validateNNoptions_edge_cases(self):
        """Test validateNNoptions edge cases"""
        # Test with None options
        with pytest.raises(ValueError):
            validateNNoptions(None)
        
        # Test with non-dict options
        with pytest.raises(ValueError):
            validateNNoptions("not_a_dict")
        
        # Test with empty dict
        result = validateNNoptions({})
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_validateNNoptions_consistency(self):
        """Test that validateNNoptions produces consistent results"""
        options = {}
        
        # Call multiple times
        result1 = validateNNoptions(options.copy())
        result2 = validateNNoptions(options.copy())
        
        # Should be consistent
        assert result1 == result2
        
        # Check that all required fields are present
        required_fields = ['method', 'order', 'taylorTerms', 'maxError', 'verbose', 'train']
        for field in required_fields:
            assert field in result1
            assert field in result2
