"""
CORAwarning - central hub for all warning messages thrown by CORA functions

This module provides the CORAwarning function that mimics MATLAB's CORAwarning
functionality for centralized warning handling in CORA.

Syntax:
    CORAwarning(identifier, category, name, version, message, *args)

Inputs:
    identifier - name of CORA warning (e.g., 'CORA:deprecated')
    category - category of warning (e.g., 'property', 'function')
    name - name of deprecated/changed item
    version - version when change was made
    message - additional information about the warning
    *args - additional arguments

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 2022 (MATLAB)
Python translation: 2025
"""

import warnings
from typing import Optional, Any


def CORAwarning(identifier: str, category: str = "", name: str = "", 
                version: str = "", message: str = "", *args) -> None:
    """
    Central hub for all warning messages thrown by CORA functions
    
    Args:
        identifier: Warning identifier (e.g., 'CORA:deprecated')
        category: Category of warning (e.g., 'property', 'function')
        name: Name of deprecated/changed item
        version: Version when change was made
        message: Additional warning message
        *args: Additional arguments
    """
    
    # Generate warning message based on identifier
    if identifier == 'CORA:deprecated':
        if category and name and version:
            warning_msg = f"Deprecated {category} '{name}' (since {version})"
            if message:
                warning_msg += f": {message}"
        else:
            warning_msg = f"Deprecated functionality: {message}"
    
    elif identifier == 'CORA:reachSet':
        warning_msg = f"Reachability analysis warning: {message}"
    
    elif identifier == 'CORA:verbose':
        warning_msg = f"Verbose mode: {message}"
    
    elif identifier == 'CORA:plotting':
        warning_msg = f"Plotting warning: {message}"
    
    else:
        # Default case
        warning_msg = f"{identifier}: {message}"
    
    # Issue the warning
    warnings.warn(warning_msg, UserWarning, stacklevel=2) 