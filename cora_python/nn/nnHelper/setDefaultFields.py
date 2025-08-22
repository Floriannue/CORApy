"""
setDefaultFields - set default fields for a given options struct.

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Lukas Koller
         Python: AI Assistant
"""

from typing import Dict, Any, List, Union, Callable


def setDefaultFields(options: Dict[str, Any], defaultFields: List[List[Any]]) -> Dict[str, Any]:
    """
    Set default fields for a given options dictionary.
    
    Args:
        options: options dictionary
        defaultFields: list containing [field_name, default_value] pairs
        
    Returns:
        options: updated options dictionary
        
    See also: validateNNoptions, validateRLoptions
    """
    # Set default value of fields if required.
    for i in range(len(defaultFields)):
        field = defaultFields[i][0]
        if field not in options:
            field_value = defaultFields[i][1]
            if callable(field_value):
                field_value = field_value(options)
            options[field] = field_value
    
    return options
