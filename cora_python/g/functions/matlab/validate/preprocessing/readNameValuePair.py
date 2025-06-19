import numpy as np
from typing import Any, Callable, List, Union, Tuple, Optional

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.checkValueAttributes import checkValueAttributes

# Assume CHECKS_ENABLED is True for now, or needs to be imported from g.macros if it exists.
# For a more robust solution, CHECKS_ENABLED would be a configuration or global variable.
CHECKS_ENABLED = True

def readNameValuePair(NVpairs: List[Any], name: str, *varargin) -> Tuple[List[Any], Any]:
    """
    readNameValuePair - searches through a list of name-value pairs to return
    the corresponding value of a given name and a redacted list (read is
    case-insensitive and works for chars and strings); additionally, the
    provided value can be checked by a function handle and one may also
    return a default value if the desired name-value pair is not given

    Syntax:
       [NVpairs,value] = readNameValuePair(NVpairs,name)
       [NVpairs,value] = readNameValuePair(NVpairs,name,check)
       [NVpairs,value] = readNameValuePair(NVpairs,name,check,def)

    Inputs:
       NVpairs - list: list of name-value pairs
       name - str: name of name-value pair
       check - (optional) list of admissible attributes
             (check function checkValueAttributes for details)
       def - (optional) default value

    Outputs:
       NVpairs - list of name-value pairs
       value - value of name-value pair

    Example:
       # NVpairs = ['Splits',8];
       # [NVpairs,value] = readNameValuePair(NVpairs,'Splits','scalar',10);

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: checkValueAttributes, inputArgsCheck

    Authors:       Mark Wetzlinger, Niklas Kochdumper, Tobias Ladner
    Written:       15-July-2020
    Last update:   24-November-2021 (allow list of check functions)
                   07-July-2022 (MW, case-insensitive, string compatibility)
                   03-March-2025 (TL, reworked using checkValueAttributes)
                   29-November-2023 (TL, check for CHECKS_ENABLED)
    Last revision: ---
    """

    def aux_formatFunc(func: Union[str, Callable]) -> str:
        if isinstance(func, Callable):
            return func.__name__  # Or use inspect.getsource for more detail
        else:
            return f"'{func}'"

    def aux_formatFuncs(funcs: Union[List[Union[str, Callable]], str, Callable]) -> str:
        if isinstance(funcs, list):
            formatted_funcs = [aux_formatFunc(f) for f in funcs]
            return f"{{'{', '.join(formatted_funcs)}'}}"
        else:
            return aux_formatFunc(funcs)

    # write function into list
    funcs = []
    if len(varargin) >= 1:
        if isinstance(varargin[0], list):
            funcs = varargin[0]
        else:
            funcs = [varargin[0]]

    # default empty value (name not found)
    value = None
    if len(varargin) >= 2:
        value = varargin[1]

    # checks enabled
    checks_enabled = CHECKS_ENABLED

    # Iterate backwards to match MATLAB's behavior when duplicates exist
    nv_pairs_copy = NVpairs[:]
    index_to_remove = -1

    for i in range(len(nv_pairs_copy) - 2, -1, -2):
        nv_name = nv_pairs_copy[i]
        if isinstance(nv_name, str) and nv_name.lower() == name.lower():
            value = nv_pairs_copy[i+1]
            index_to_remove = i
            break

    if index_to_remove != -1:
        # Name found, remove the pair
        NVpairs.pop(index_to_remove)
        NVpairs.pop(index_to_remove) # Pop the value as well
        
        if checks_enabled and funcs:
            # check whether name complies with check
            res = checkValueAttributes(value, '', funcs) # class_name empty to only check attributes
            if not res:
                raise CORAerror('CORA:specialError',
                                f"Invalid assignment for name-value pair '{name}': Must pass {aux_formatFuncs(funcs)}.")

    return NVpairs, value 