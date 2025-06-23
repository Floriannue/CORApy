"""
setDefaultValues - alias for set_default_values to match MATLAB naming

This is a simple wrapper around set_default_values to maintain compatibility
with MATLAB function naming conventions.

Syntax:
    values = setDefaultValues(defaults, args)

Inputs:
    defaults - list of default values  
    args - input arguments list

Outputs:
    values - processed values with defaults applied

Authors: Python translation by AI Assistant
Written: 2025
"""

from .set_default_values import set_default_values


# Simple alias for MATLAB compatibility
setDefaultValues = set_default_values 