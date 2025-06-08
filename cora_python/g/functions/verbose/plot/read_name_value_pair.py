"""
read_name_value_pair - reads name-value pairs from argument list

This function mimics MATLAB's readNameValuePair functionality for extracting
specific name-value pairs from argument lists.

Syntax:
    remaining_args, value = read_name_value_pair(args, name, validator=None, default=None)

Inputs:
    args - list of arguments containing name-value pairs
    name - name of the parameter to extract
    validator - validation function or criteria (optional)
    default - default value if not found (optional)

Outputs:
    remaining_args - argument list with extracted pair removed
    value - extracted value or default

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import List, Any, Optional, Union, Callable


def read_name_value_pair(args: List[Any], name: str, validator: Optional[Union[str, List[str], Callable]] = None, 
                        default: Any = None) -> tuple[List[Any], Any]:
    """
    Read and extract a name-value pair from argument list
    
    Args:
        args: List of arguments containing name-value pairs
        name: Name of the parameter to extract
        validator: Validation criteria ('isscalar', 'islogical', etc.) or function
        default: Default value if not found
        
    Returns:
        Tuple of (remaining_args, extracted_value)
    """
    remaining_args = args.copy()
    value = default
    
    # Find the name in the argument list
    for i in range(0, len(args) - 1, 2):
        if isinstance(args[i], str) and args[i] == name:
            # Found the name-value pair
            value = args[i + 1]
            
            # Validate if validator provided
            if validator is not None:
                _validate_value(value, validator)
            
            # Remove the name-value pair from remaining args
            remaining_args = args[:i] + args[i+2:]
            break
    
    return remaining_args, value


def _validate_value(value: Any, validator: Union[str, List[str], Callable]) -> None:
    """Validate extracted value"""
    if isinstance(validator, str):
        _validate_single_criterion(value, validator)
    elif isinstance(validator, list):
        for criterion in validator:
            _validate_single_criterion(value, criterion)
    elif callable(validator):
        if not validator(value):
            raise ValueError(f"Value {value} failed custom validation")


def _validate_single_criterion(value: Any, criterion: str) -> None:
    """Validate against single criterion"""
    if criterion == 'isscalar':
        if not np.isscalar(value) and not (isinstance(value, np.ndarray) and value.size == 1):
            raise ValueError(f"Value must be scalar, got {type(value)}")
            
    elif criterion == 'islogical':
        if not isinstance(value, bool) and not (isinstance(value, np.ndarray) and value.dtype == bool):
            raise ValueError(f"Value must be logical/boolean, got {type(value)}")
            
    elif criterion == 'isnumeric':
        if not isinstance(value, (int, float, complex, np.number)) and \
           not (isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)):
            raise ValueError(f"Value must be numeric, got {type(value)}")
            
    elif criterion == 'ischar':
        if not isinstance(value, str):
            raise ValueError(f"Value must be string/char, got {type(value)}")
            
    elif criterion == 'isempty':
        if isinstance(value, np.ndarray) and value.size > 0:
            raise ValueError("Value must be empty")
        elif value is not None and value != []:
            raise ValueError("Value must be empty")
            
    else:
        # Could be a specific value check or other criterion
        pass 