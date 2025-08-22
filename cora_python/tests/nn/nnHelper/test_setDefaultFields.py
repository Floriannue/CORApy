"""
Test for nnHelper.setDefaultFields function

This test verifies that the setDefaultFields function works correctly for setting default values.
"""

import pytest
import numpy as np
from cora_python.nn.nnHelper.setDefaultFields import setDefaultFields


class TestSetDefaultFields:
    """Test class for setDefaultFields function"""
    
    def test_setDefaultFields_basic(self):
        """Test basic setDefaultFields functionality"""
        # Test with simple defaults
        options = {}
        default_fields = [
            ['field1', 'value1'],
            ['field2', 42],
            ['field3', True]
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Check that all default fields are set
        assert result['field1'] == 'value1'
        assert result['field2'] == 42
        assert result['field3'] == True
        
        # Check that original options dict is modified
        assert options['field1'] == 'value1'
        assert options['field2'] == 42
        assert options['field3'] == True
    
    def test_setDefaultFields_existing_values(self):
        """Test that existing values are not overwritten"""
        # Test with existing values
        options = {
            'field1': 'existing_value',
            'field2': 100
        }
        default_fields = [
            ['field1', 'default_value'],
            ['field2', 42],
            ['field3', True]
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Check that existing values are preserved
        assert result['field1'] == 'existing_value'
        assert result['field2'] == 100
        
        # Check that new default is set
        assert result['field3'] == True
        
        # Check that original options dict is modified
        assert options['field1'] == 'existing_value'
        assert options['field2'] == 100
        assert options['field3'] == True
    
    def test_setDefaultFields_callable_defaults(self):
        """Test setDefaultFields with callable default values"""
        options = {}
        default_fields = [
            ['field1', lambda opts: 'computed_value'],
            ['field2', lambda opts: len(opts) + 1],
            ['field3', lambda opts: {'computed': True, 'count': len(opts)}]
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Check that callable defaults are computed
        assert result['field1'] == 'computed_value'
        assert result['field2'] == 1  # len({}) + 1
        assert result['field3'] == {'computed': True, 'count': 0}
        
        # Check that original options dict is modified
        assert options['field1'] == 'computed_value'
        assert options['field2'] == 1
        assert options['field3'] == {'computed': True, 'count': 0}
    
    def test_setDefaultFields_callable_with_existing_options(self):
        """Test callable defaults that depend on existing options"""
        options = {
            'existing_field': 'test_value',
            'count': 5
        }
        default_fields = [
            ['field1', lambda opts: opts.get('existing_field', 'default')],
            ['field2', lambda opts: opts.get('count', 0) * 2],
            ['field3', lambda opts: f"computed_{opts.get('count', 0)}"]
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Check that callable defaults use existing options
        assert result['field1'] == 'test_value'
        assert result['field2'] == 10  # 5 * 2
        assert result['field3'] == 'computed_5'
        
        # Check that original options dict is modified
        assert options['field1'] == 'test_value'
        assert options['field2'] == 10
        assert options['field3'] == 'computed_5'
    
    def test_setDefaultFields_empty_defaults(self):
        """Test setDefaultFields with empty default fields"""
        options = {'existing': 'value'}
        default_fields = []
        
        result = setDefaultFields(options, default_fields)
        
        # Should return options unchanged
        assert result == options
        assert result['existing'] == 'value'
    
    def test_setDefaultFields_empty_options(self):
        """Test setDefaultFields with empty options"""
        options = {}
        default_fields = [
            ['field1', 'value1'],
            ['field2', 42]
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Should set all defaults
        assert result['field1'] == 'value1'
        assert result['field2'] == 42
        
        # Should be the same object
        assert result is options
    
    def test_setDefaultFields_mixed_types(self):
        """Test setDefaultFields with mixed types"""
        options = {}
        default_fields = [
            ['string_field', 'string_value'],
            ['int_field', 42],
            ['float_field', 3.14],
            ['bool_field', True],
            ['list_field', [1, 2, 3]],
            ['dict_field', {'key': 'value'}],
            ['none_field', None],
            ['array_field', np.array([1, 2, 3])]
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Check all types
        assert isinstance(result['string_field'], str)
        assert isinstance(result['int_field'], int)
        assert isinstance(result['float_field'], float)
        assert isinstance(result['bool_field'], bool)
        assert isinstance(result['list_field'], list)
        assert isinstance(result['dict_field'], dict)
        assert result['none_field'] is None
        assert isinstance(result['array_field'], np.ndarray)
        
        # Check values
        assert result['string_field'] == 'string_value'
        assert result['int_field'] == 42
        assert result['float_field'] == 3.14
        assert result['bool_field'] == True
        assert result['list_field'] == [1, 2, 3]
        assert result['dict_field'] == {'key': 'value'}
        assert result['none_field'] is None
        assert np.array_equal(result['array_field'], np.array([1, 2, 3]))
    
    def test_setDefaultFields_nested_callables(self):
        """Test setDefaultFields with nested callable structures"""
        options = {}
        default_fields = [
            ['nested_dict', lambda opts: {
                'inner_field': lambda inner_opts: 'inner_value',
                'computed_field': lambda inner_opts: len(inner_opts) + 1
            }],
            ['nested_list', lambda opts: [
                lambda list_opts: 'first',
                lambda list_opts: 'second',
                lambda list_opts: len(list_opts)
            ]]
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Check nested structure
        assert isinstance(result['nested_dict'], dict)
        assert callable(result['nested_dict']['inner_field'])
        assert callable(result['nested_dict']['computed_field'])
        
        assert isinstance(result['nested_list'], list)
        assert all(callable(item) for item in result['nested_list'])
    
    def test_setDefaultFields_callable_side_effects(self):
        """Test that callable defaults can modify options during computation"""
        options = {}
        default_fields = [
            ['field1', lambda opts: opts.update({'computed': True}) or 'value1'],
            ['field2', lambda opts: opts.get('computed', False)]
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Check that side effects work
        assert result['field1'] == 'value1'
        assert result['field2'] == True
        assert result['computed'] == True
        
        # Check that original options dict is modified
        assert options['field1'] == 'value1'
        assert options['field2'] == True
        assert options['computed'] == True
    
    def test_setDefaultFields_error_handling(self):
        """Test setDefaultFields error handling"""
        options = {}
        
        # Test with invalid default_fields format
        invalid_defaults = [
            ['field1'],  # Missing default value
            ['field2', 'value2', 'extra'],  # Too many elements
            [],  # Empty element
            'not_a_list'  # Not a list
        ]
        
        # Should handle gracefully or raise appropriate error
        try:
            result = setDefaultFields(options, invalid_defaults)
        except Exception as e:
            # Error is acceptable for invalid input
            pass
    
    def test_setDefaultFields_consistency(self):
        """Test that setDefaultFields produces consistent results"""
        options = {}
        default_fields = [
            ['field1', 'value1'],
            ['field2', lambda opts: 'computed_value'],
            ['field3', 42]
        ]
        
        # Call multiple times
        result1 = setDefaultFields(options.copy(), default_fields)
        result2 = setDefaultFields(options.copy(), default_fields)
        
        # Should be consistent
        assert result1['field1'] == result2['field1']
        assert result1['field2'] == result2['field2']
        assert result1['field3'] == result2['field3']
    
    def test_setDefaultFields_complex_scenarios(self):
        """Test setDefaultFields with complex scenarios"""
        # Test with options that have nested structure
        options = {
            'nested': {
                'existing': 'value'
            }
        }
        default_fields = [
            ['nested', lambda opts: {
                'existing': opts.get('nested', {}).get('existing', 'default'),
                'new_field': 'new_value'
            }],
            ['top_level', 'top_value']
        ]
        
        result = setDefaultFields(options, default_fields)
        
        # Check nested structure
        assert result['nested']['existing'] == 'value'
        assert result['nested']['new_field'] == 'new_value'
        assert result['top_level'] == 'top_value'
        
        # Check that original options dict is modified
        assert options['nested']['existing'] == 'value'
        assert options['nested']['new_field'] == 'new_value'
        assert options['top_level'] == 'top_value'
    
    def test_setDefaultFields_performance(self):
        """Test setDefaultFields performance with many fields"""
        options = {}
        default_fields = [
            [f'field_{i}', f'value_{i}'] for i in range(100)
        ]
        
        # Should handle many fields efficiently
        result = setDefaultFields(options, default_fields)
        
        # Check that all fields are set
        for i in range(100):
            assert result[f'field_{i}'] == f'value_{i}'
        
        # Check that original options dict is modified
        for i in range(100):
            assert options[f'field_{i}'] == f'value_{i}'
