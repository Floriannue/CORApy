"""
Test for nnHelper.validateRLoptions function

This test verifies that the validateRLoptions function works correctly for RL options validation.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.validateRLoptions import validateRLoptions


class TestValidateRLoptions:
    """Test class for validateRLoptions function"""
    
    def test_validateRLoptions_empty(self):
        """Test validateRLoptions with empty options"""
        options = {}
        result = validateRLoptions(options)
        
        # Should not raise error and return options with defaults
        assert isinstance(result, dict)
        assert 'rl' in result
        assert 'actor' in result['rl']
        assert 'critic' in result['rl']
    
    def test_validateRLoptions_default_values(self):
        """Test that default values are set correctly"""
        options = {}
        result = validateRLoptions(options)
        
        # Check RL defaults
        rl = result['rl']
        assert rl['gamma'] == 0.99
        assert rl['tau'] == 0.005
        assert rl['expNoise'] == 0.2
        assert rl['expNoiseTarget'] == 0.2
        assert rl['expNoiseType'] == 'OU'
        assert rl['expDecayFactor'] == 1
        assert rl['batchsize'] == 64
        assert rl['buffersize'] == 1e6
        assert rl['noise'] == 0.1
        assert rl['earlyStop'] == float('inf')
        assert rl['printFreq'] == 50
        assert rl['visRate'] == 50
    
    def test_validateRLoptions_actor_defaults(self):
        """Test that actor training defaults are set correctly"""
        options = {}
        result = validateRLoptions(options)
        
        actor_train = result['rl']['actor']['nn']['train']
        assert actor_train['eta'] == 0.1
        assert actor_train['omega'] == 0.5
        assert actor_train['exact_backprop'] == False
        assert actor_train['zonotope_weight_update'] == 'outer_product'
        
        # Check optimizer structure
        assert 'type' in actor_train['optim']
        assert actor_train['optim']['type'] == 'Adam'
        assert actor_train['optim']['lr'] == 1e-4
        assert actor_train['optim']['beta1'] == 0.9
        assert actor_train['optim']['beta2'] == 0.999
        assert actor_train['optim']['eps'] == 1e-8
        assert actor_train['optim']['weight_decay'] == 0
    
    def test_validateRLoptions_critic_defaults(self):
        """Test that critic training defaults are set correctly"""
        options = {}
        result = validateRLoptions(options)
        
        critic_train = result['rl']['critic']['nn']['train']
        assert critic_train['eta'] == 0.01
        assert critic_train['exact_backprop'] == False
        assert critic_train['zonotope_weight_update'] == 'outer_product'
        
        # Check optimizer structure
        assert 'type' in critic_train['optim']
        assert critic_train['optim']['type'] == 'Adam'
        assert critic_train['optim']['lr'] == 1e-3
        assert critic_train['optim']['beta1'] == 0.9
        assert critic_train['optim']['beta2'] == 0.999
        assert critic_train['optim']['eps'] == 1e-8
        assert critic_train['optim']['weight_decay'] == 1e-2
    
    def test_validateRLoptions_adversarial_defaults(self):
        """Test that adversarial operation defaults are set correctly"""
        options = {}
        result = validateRLoptions(options)
        
        adv_ops = result['rl']['actor']['nn']['train']['advOps']
        assert adv_ops['numSamples'] == 200
        assert adv_ops['alpha'] == 4
        assert adv_ops['beta'] == 4
    
    def test_validateRLoptions_custom_values(self):
        """Test that custom values override defaults"""
        options = {
            'rl': {
                'gamma': 0.95,
                'tau': 0.01,
                'actor': {
                    'nn': {
                        'train': {
                            'eta': 0.2,
                            'omega': 0.7
                        }
                    }
                },
                'critic': {
                    'nn': {
                        'train': {
                            'eta': 0.05
                        }
                    }
                }
            }
        }
        
        result = validateRLoptions(options)
        
        # Check that custom values are preserved
        assert result['rl']['gamma'] == 0.95
        assert result['rl']['tau'] == 0.01
        assert result['rl']['actor']['nn']['train']['eta'] == 0.2
        assert result['rl']['actor']['nn']['train']['omega'] == 0.7
        assert result['rl']['critic']['nn']['train']['eta'] == 0.05
        
        # Check that other defaults are still set
        assert result['rl']['expNoise'] == 0.2
        assert result['rl']['batchsize'] == 64
    
    def test_validateRLoptions_nested_structure(self):
        """Test that nested structure is created correctly"""
        options = {}
        result = validateRLoptions(options)
        
        # Check that all required nested structures exist
        assert 'rl' in result
        assert 'actor' in result['rl']
        assert 'nn' in result['rl']['actor']
        assert 'train' in result['rl']['actor']['nn']
        assert 'critic' in result['rl']
        assert 'nn' in result['rl']['critic']
        assert 'train' in result['rl']['critic']['nn']
        assert 'advOps' in result['rl']['actor']['nn']['train']
    
    def test_validateRLoptions_validation_checks(self):
        """Test that validation checks work correctly"""
        options = {}
        result = validateRLoptions(options)
        
        # Test valid values
        valid_options = result.copy()
        valid_options['rl']['gamma'] = 0.5
        valid_options['rl']['tau'] = 0.1
        valid_options['rl']['expNoiseType'] = 'gaussian'
        
        # Should not raise error
        try:
            validateRLoptions(valid_options)
        except Exception as e:
            pytest.fail(f"Valid options should not raise error: {e}")
    
    def test_validateRLoptions_invalid_values(self):
        """Test that invalid values raise appropriate errors"""
        options = {}
        result = validateRLoptions(options)
        
        # Test invalid gamma (should be in [0, 1])
        invalid_options = result.copy()
        invalid_options['rl']['gamma'] = 1.5
        
        with pytest.raises(ValueError, match="gamma.*outside the valid domain"):
            validateRLoptions(invalid_options)
        
        # Test invalid tau (should be in [0, 1])
        invalid_options = result.copy()
        invalid_options['rl']['tau'] = -0.1
        
        with pytest.raises(ValueError, match="tau.*outside the valid domain"):
            validateRLoptions(invalid_options)
        
        # Test invalid expNoiseType
        invalid_options = result.copy()
        invalid_options['rl']['expNoiseType'] = 'invalid_type'
        
        with pytest.raises(ValueError, match="expNoiseType.*invalid value"):
            validateRLoptions(invalid_options)
    
    def test_validateRLoptions_training_method_consistency(self):
        """Test training method consistency between actor and critic"""
        options = {}
        result = validateRLoptions(options)
        
        # Set critic method to 'set' but actor method to 'point'
        invalid_options = result.copy()
        invalid_options['rl']['critic']['nn']['train']['method'] = 'set'
        invalid_options['rl']['actor']['nn']['train']['method'] = 'point'
        
        with pytest.raises(ValueError, match="critic.nn.train.method cannot be 'set' when actor.nn.train.method is not 'set'"):
            validateRLoptions(invalid_options)
        
        # Set both to 'set' - should work
        valid_options = result.copy()
        valid_options['rl']['critic']['nn']['train']['method'] = 'set'
        valid_options['rl']['actor']['nn']['train']['method'] = 'set'
        
        try:
            validateRLoptions(valid_options)
        except Exception as e:
            pytest.fail(f"Valid training method combination should not raise error: {e}")
    
    def test_validateRLoptions_edge_cases(self):
        """Test validateRLoptions edge cases"""
        # Test with None values
        options = {'rl': None}
        
        with pytest.raises(KeyError):
            validateRLoptions(options)
        
        # Test with missing nested structures
        options = {'rl': {}}
        
        with pytest.raises(KeyError):
            validateRLoptions(options)
    
    def test_validateRLoptions_reuse_options(self):
        """Test that options can be reused and modified"""
        options = {}
        
        # First call
        result1 = validateRLoptions(options)
        assert result1['rl']['gamma'] == 0.99
        
        # Modify and call again
        result1['rl']['gamma'] = 0.8
        result2 = validateRLoptions(result1)
        
        # Should preserve the modified value
        assert result2['rl']['gamma'] == 0.8
        
        # Should still have all other defaults
        assert result2['rl']['tau'] == 0.005
        assert result2['rl']['batchsize'] == 64
    
    def test_validateRLoptions_complete_validation(self):
        """Test complete validation of all fields"""
        options = {}
        result = validateRLoptions(options)
        
        # Test all numeric fields with valid ranges
        test_cases = [
            ('gamma', (0, 1)),
            ('tau', (0, 1)),
            ('expNoise', (0, float('inf'))),
            ('expNoiseTarget', (0, float('inf'))),
            ('expDecayFactor', (-1, 1)),
            ('batchsize', (0, float('inf'))),
            ('noise', (0, float('inf')))
        ]
        
        for field, (min_val, max_val) in test_cases:
            # Test minimum value
            test_options = result.copy()
            test_options['rl'][field] = min_val
            try:
                validateRLoptions(test_options)
            except Exception as e:
                pytest.fail(f"Field {field} with value {min_val} should be valid: {e}")
            
            # Test maximum value
            test_options = result.copy()
            test_options['rl'][field] = max_val
            try:
                validateRLoptions(test_options)
            except Exception as e:
                pytest.fail(f"Field {field} with value {max_val} should be valid: {e}")
    
    def test_validateRLoptions_string_validation(self):
        """Test string field validation"""
        options = {}
        result = validateRLoptions(options)
        
        # Test valid expNoiseType values
        valid_types = ['OU', 'gaussian']
        for noise_type in valid_types:
            test_options = result.copy()
            test_options['rl']['expNoiseType'] = noise_type
            try:
                validateRLoptions(test_options)
            except Exception as e:
                pytest.fail(f"expNoiseType '{noise_type}' should be valid: {e}")
        
        # Test valid training method values
        valid_methods = ['point', 'set']
        for method in valid_methods:
            test_options = result.copy()
            test_options['rl']['critic']['nn']['train']['method'] = method
            test_options['rl']['actor']['nn']['train']['method'] = method
            try:
                validateRLoptions(test_options)
            except Exception as e:
                pytest.fail(f"Training method '{method}' should be valid: {e}")
