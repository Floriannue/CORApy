import numpy as np
from typing import Any

def _get_dim(obj: Any):
    if hasattr(obj, 'dim'):
        val = obj.dim() if callable(obj.dim) else obj.dim
        if isinstance(val, list):
            return tuple(val)
        return (val,)  # a single dimension
    if hasattr(obj, 'shape'):
        return obj.shape
    if isinstance(obj, (int, float)):
        return (1,)
    return ()

def _is_instance_by_name(obj: Any, name: str) -> bool:
    if hasattr(obj, '__class__'):
        for c in obj.__class__.__mro__:
            if c.__name__ == name:
                return True
    return False

def equal_dim_check(s1: Any, s2: Any, return_value: bool = False) -> bool:
    
    res = True
    
    dim1 = _get_dim(s1)
    dim2 = _get_dim(s2)

    is_s1_numeric = isinstance(s1, (np.ndarray, int, float))
    is_s2_numeric = isinstance(s2, (np.ndarray, int, float))
    is_s1_cont_set = _is_instance_by_name(s1, 'ContSet')
    is_s2_cont_set = _is_instance_by_name(s2, 'ContSet')
    is_s1_matrix_set = _is_instance_by_name(s1, 'MatrixSet')

    if is_s1_cont_set and is_s2_cont_set:
        if dim1 != dim2:
            res = False
    elif is_s1_cont_set and is_s2_numeric:
        # operation between set and matrix/vector/scalar
        # row dimension of matrix has to fit set dimension (MATLAB logic)
        if not np.isscalar(s2):
            if len(dim1) == 1:
                # S1 is regular set
                if dim1[0] != dim2[0]:
                    res = False
            else:
                # S1 is matrix set, allow broadcasting (not implemented yet)
                # For now, use simple comparison
                if dim1 != dim2:
                    res = False
    elif (is_s1_numeric or is_s1_matrix_set) and is_s2_cont_set:
        # operation between matrix/matrix set and contSet
        # column dimension of matrix has to fit set dimension
        if not np.isscalar(s1) and len(dim1) > 1 and len(dim2) > 0 and dim1[1] != dim2[0]:
            res = False
    
    if not return_value and not res:
        from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
        raise CORAerror('CORA:dimensionMismatch', s1, s2)
        
    return res 