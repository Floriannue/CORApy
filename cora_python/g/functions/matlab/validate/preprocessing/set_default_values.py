"""
set_default_values - sets default values for function arguments

This function mimics MATLAB's setDefaultValues functionality for handling
optional arguments and setting defaults.

Syntax:
    values = set_default_values(defaults, args)

Inputs:
    defaults - list of default values  
    args - input arguments list

Outputs:
    values - processed values with defaults applied

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 2020 (MATLAB)
Python translation: 2025
"""

from typing import List, Any, Optional, Tuple


def set_default_values(defaults: List[Any], *args: Any) -> List[Any]:
    """
    Set default values for function arguments (matches MATLAB exactly)
    
    Args:
        defaults: List of default values
        *args: Input arguments list (can be None or empty)
        
    Returns:
        List of processed values with defaults applied (matches MATLAB varargout)
    """
    # Ensure defaults is a list to allow item assignment
    defaults_list = list(defaults)
    result = defaults_list.copy()
    
    # Flatten args in case they are passed as a list/tuple
    args_list = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            args_list.extend(arg)
        else:
            args_list.append(arg)
            
    # Override with provided arguments (matches MATLAB exactly)
    num_given = len(args_list)
    num_defaults = len(defaults_list)
    
    # assign default values if corresponding values are not provided
    # matches MATLAB: varargout = [givenValues(1:n_given), defaultValues(n_given+1:n_default)];
    for i in range(num_defaults):
        if i < num_given:
            result[i] = args_list[i]
        # else: keep default value (already in result)
    
    return result 