"""
rmiffield - removes a field from a struct if exists

Syntax:
    list = rmiffield(list,field)

Inputs:
    list - struct (dict)
    field - field name (list of strings, or string)

Outputs:
    list - struct (dict)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner, Mark Wetzlinger
Written:       17-October-2023
Last update:   30-August-2024 (MW, support cell-array of fields)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Union, Dict, Any, List


def rmiffield(list_dict: Dict[str, Any], field: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Removes a field from a struct (dict) if it exists
    
    Args:
        list_dict: struct (dict)
        field: field name (list of strings, or string)
        
    Returns:
        list_dict: struct (dict) with field(s) removed
    """
    
    # MATLAB: if iscell(field)
    if isinstance(field, list):
        # cell array of fields
        # MATLAB: for i=1:length(field)
        for i in range(len(field)):
            # MATLAB: if isfield(list,field{i})
            if field[i] in list_dict:
                # MATLAB: list = rmfield(list,field{i});
                list_dict = {k: v for k, v in list_dict.items() if k != field[i]}
    else:
        # single field
        # MATLAB: if isfield(list,field)
        if field in list_dict:
            # MATLAB: list = rmfield(list,field);
            list_dict = {k: v for k, v in list_dict.items() if k != field}
    
    return list_dict

