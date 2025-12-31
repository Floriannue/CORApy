"""
getDefaultValue - contains list of default values for params / options

Syntax:
    defValue = getDefaultValue(field,sys,params,options,listname)

Inputs:
    field - struct field in params / options
    sys - object of system class
    params - struct containing model parameters
    options - struct containing algorithm parameters
    listname - 'params' or 'options'

Outputs:
    defValue - default value for given field

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: getDefaultValuesParams, getDefaultValuesOptions

Authors:       Mark Wetzlinger
Written:       26-January-2021
Last update:   09-October-2023 (TL, split options/params)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Dict
from cora_python.g.functions.helper.dynamics.checkOptions.getDefaultValueOptions import getDefaultValueOptions
from cora_python.g.functions.helper.dynamics.checkOptions.getDefaultValueParams import getDefaultValueParams
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def getDefaultValue(field: str, sys: Any, params: Dict[str, Any], 
                    options: Dict[str, Any], listname: str) -> Any:
    """
    Get default value for a field in params or options
    
    Args:
        field: struct field in params / options
        sys: object of system class
        params: struct containing model parameters
        options: struct containing algorithm parameters
        listname: 'params' or 'options'
        
    Returns:
        defValue: default value for given field
    """
    
    # split list for params / options for more transparency / readability
    # MATLAB: switch listname
    if listname == 'params':
        # search for default value in params
        # MATLAB: defValue = getDefaultValueParams(field,sys,params,options);
        defValue = getDefaultValueParams(field, sys, params, options)
    
    elif listname == 'options':
        # search for default value in options
        # MATLAB: defValue = getDefaultValueOptions(field,sys,params,options);
        defValue = getDefaultValueOptions(field, sys, params, options)
    
    else:
        # MATLAB: throw(CORAerror('CORA:specialError',sprintf('Unknown list name: %s', listname)))
        raise CORAerror('CORA:specialError', f'Unknown list name: {listname}')
    
    return defValue

