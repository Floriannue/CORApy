"""
NVpairsPlainSetDefaultValues - set default values for name-value pairs (simple version)

Syntax:
    values = NVpairsPlainSetDefaultValues(listOfNameValuePairs, argNames, argValues)

Inputs:
    listOfNameValuePairs - list of default name-value pairs [name1, value1, name2, value2, ...]
    argNames - list of argument names provided by user
    argValues - list of argument values provided by user

Outputs:
    values - list of final values in the order of listOfNameValuePairs

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: readNameValuePair

Authors: Python translation by AI Assistant
Written: 2025
"""

from typing import List, Any, Union


def NVpairsPlainSetDefaultValues(listOfNameValuePairs: List[Any], 
                                argNames: List[str], 
                                argValues: List[Any]) -> List[Any]:
    """
    Set default values for name-value pairs (simple version)
    
    This function processes name-value pairs and returns the final values,
    using user-provided values where available and defaults otherwise.
    
    Args:
        listOfNameValuePairs: list of default name-value pairs [name1, value1, name2, value2, ...]
        argNames: list of argument names provided by user
        argValues: list of argument values provided by user
        
    Returns:
        values: list of final values in the order of listOfNameValuePairs
    """
    
    # Parse default name-value pairs into a dictionary
    defaults = {}
    for i in range(0, len(listOfNameValuePairs), 2):
        if i + 1 < len(listOfNameValuePairs):
            name = listOfNameValuePairs[i]
            value = listOfNameValuePairs[i + 1]
            defaults[name] = value
    
    # Create dictionary of user-provided values
    user_values = {}
    for i, name in enumerate(argNames):
        if i < len(argValues):
            user_values[name] = argValues[i]
    
    # Build result list in order of defaults
    result = []
    for i in range(0, len(listOfNameValuePairs), 2):
        if i + 1 < len(listOfNameValuePairs):
            name = listOfNameValuePairs[i]
            if name in user_values:
                result.append(user_values[name])
            else:
                result.append(listOfNameValuePairs[i + 1])
    
    return result 