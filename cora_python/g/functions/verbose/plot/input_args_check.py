"""
input_args_check - validates input arguments

This function mimics MATLAB's inputArgsCheck functionality for argument validation.

Syntax:
    input_args_check(checks)

Inputs:
    checks - list of validation specifications

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import List, Any, Union, Callable
from ....functions.matlab.validate.postprocessing.CORAerror import CORAError


def input_args_check(checks: List[List[Any]]) -> None:
    """
    Validate input arguments
    
    Args:
        checks: List of validation specifications
               Each check is [value, type_check, additional_checks]
    """
    for check in checks:
        if len(check) < 2:
            continue
            
        value = check[0]
        type_check = check[1]
        additional_checks = check[2:] if len(check) > 2 else []
        
        # Basic type checking
        if type_check == 'att':
            # Attribute-based type checking
            if len(additional_checks) > 0:
                expected_type = additional_checks[0]
                _validate_type(value, expected_type)
                
        elif type_check == 'str':
            # String validation
            if not isinstance(value, str):
                raise CORAError('CORA:wrongValue', f'Expected string, got {type(value)}')
            if len(additional_checks) > 0:
                valid_values = additional_checks[0]
                if isinstance(valid_values, list) and value not in valid_values:
                    raise CORAError('CORA:wrongValue', f'Value must be one of {valid_values}')
                    
        elif type_check == 'numeric':
            # Numeric validation
            if not _is_numeric(value):
                raise CORAError('CORA:wrongValue', f'Expected numeric value, got {type(value)}')
            _validate_numeric_properties(value, additional_checks)


def _validate_type(value: Any, expected_type: str) -> None:
    """Validate object type"""
    if expected_type == 'contSet':
        from .....contSet.contSet.contSet import ContSet
        if not isinstance(value, ContSet):
            raise CORAError('CORA:wrongValue', f'Expected contSet object, got {type(value)}')
    elif expected_type == 'numeric':
        if not _is_numeric(value):
            raise CORAError('CORA:wrongValue', f'Expected numeric value, got {type(value)}')


def _is_numeric(value: Any) -> bool:
    """Check if value is numeric"""
    if isinstance(value, (int, float, complex, np.number)):
        return True
    elif isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.number)
    elif isinstance(value, (list, tuple)):
        # Check if all elements are numeric
        try:
            return all(_is_numeric(item) for item in value)
        except:
            return False
    else:
        return False


def _validate_numeric_properties(value: Any, properties: List[Union[str, dict]]) -> None:
    """Validate numeric properties"""
    for prop in properties:
        if isinstance(prop, dict):
            for key, val in prop.items():
                _check_numeric_property(value, key, val)
        elif isinstance(prop, str):
            _check_numeric_property(value, prop, True)


def _check_numeric_property(value: Any, prop: str, expected: Any) -> None:
    """Check specific numeric property"""
    if prop == 'nonempty':
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise CORAError('CORA:wrongValue', 'Value must not be empty')
        elif not value:
            raise CORAError('CORA:wrongValue', 'Value must not be empty')
            
    elif prop == 'integer':
        if isinstance(value, np.ndarray):
            if not np.all(np.equal(np.mod(value, 1), 0)):
                raise CORAError('CORA:wrongValue', 'Value must be integer')
        elif not isinstance(value, int):
            raise CORAError('CORA:wrongValue', 'Value must be integer')
            
    elif prop == 'positive':
        if isinstance(value, np.ndarray):
            if not np.all(value > 0):
                raise CORAError('CORA:wrongValue', 'Value must be positive')
        elif value <= 0:
            raise CORAError('CORA:wrongValue', 'Value must be positive')
            
    elif prop == 'vector':
        if isinstance(value, np.ndarray):
            if value.ndim > 2 or (value.ndim == 2 and min(value.shape) > 1):
                raise CORAError('CORA:wrongValue', 'Value must be a vector')
        elif not hasattr(value, '__len__'):
            # Scalar values are considered vectors of length 1
            pass 