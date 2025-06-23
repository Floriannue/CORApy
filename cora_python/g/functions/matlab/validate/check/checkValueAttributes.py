import numpy as np
from typing import Any, Callable, List, Union

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def checkValueAttributes(value: Any, class_name: str, attributes: List[Union[str, Callable]]) -> bool:
    """
    checkValueAttributes - checks if the given value is of the correct class
    and has the correct attributes

    Syntax:
       res = checkValueAttributes(value,class,attributes)

    Inputs:
       value - value to be tested
       class_name - string, class of the value
       attributes - list of attributes (should evaluate to logical),
                  either
                  - function_handle - takes value as input and returns logical
                  - string, such that it can be evaluated
                              via feval or custom auxiliary function

    Outputs:
       res - logical

    Example:
       # value = 1;
       # res = checkValueAttributes(value, 'numeric', ['integer','nonnan', lambda v: v >= 1])

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: inputArgsCheck, readNameValuePair

    Authors:       Tobias Ladner
    Written:       03-March-2025
    Last update:   ---
    Last revision: ---
    """

    # Auxiliary functions (nested to encapsulate logic, or could be separate private functions)
    def aux_isnd(val: Any, n: int) -> bool:
        return val.ndim == n if hasattr(val, 'ndim') else False

    def aux_issquare(val: Any) -> bool:
        if hasattr(val, 'shape') and len(val.shape) == 2:
            return val.shape[0] == val.shape[1]
        return False

    def aux_isbinary(val: Any) -> bool:
        return np.all((val == 0) | (val == 1))

    def aux_isinteger(val: Any) -> bool:
        return np.all(val == np.floor(val))

    def aux_iseven(val: Any) -> bool:
        return np.all(val % 2 == 0)

    def aux_isodd(val: Any) -> bool:
        return np.all(val % 2 != 0)

    def aux_ispositive(val: Any) -> bool:
        return np.all(val > 0)

    def aux_isnegative(val: Any) -> bool:
        return np.all(val < 0)

    def aux_iszero(val: Any) -> bool:
        return np.all(val == 0)

    def aux_checkAttribute(val: Any, cls_name: str, attribute: Union[str, Callable], reduction: str) -> bool:
        res = False
        if isinstance(attribute, str):
            # check negation
            if attribute.startswith('non'):
                negated_attribute = attribute[3:]
                res = not aux_checkAttribute(val, cls_name, negated_attribute, 'any')
                return res

            # evaluate custom attributes
            if attribute in ['2d', 'is2d']:
                res = aux_isnd(val, 2)
            elif attribute in ['3d', 'is3d']:
                res = aux_isnd(val, 3)
            elif attribute in ['square', 'issquare']:
                res = aux_issquare(val)
            elif attribute in ['binary', 'isbinary']:
                res = aux_isbinary(val)
            elif attribute in ['integer', 'isinteger']:
                res = aux_isinteger(val)
            elif attribute in ['even', 'iseven']:
                res = aux_iseven(val)
            elif attribute in ['odd', 'isodd']:
                res = aux_isodd(val)
            elif attribute in ['negative', 'isnegative']:
                res = aux_isnegative(val)
            elif attribute in ['positive', 'ispositive']:
                res = aux_ispositive(val)
            elif attribute in ['zero', 'iszero']:
                res = aux_iszero(val)
            elif attribute == 'scalar':
                res = np.isscalar(val)
            elif attribute == 'row':
                res = (isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] == 1)
            elif attribute == 'column':
                res = (isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == 1)
            elif attribute == 'vector':
                res = (isinstance(val, np.ndarray) and val.ndim == 2 and (val.shape[0] == 1 or val.shape[1] == 1))
            elif attribute == 'real':
                res = np.isreal(val).all()
            elif attribute == 'finite':
                res = np.isfinite(val).all()
            elif attribute == 'nonnan':
                res = (not np.isnan(val).any()) if isinstance(val, np.ndarray) else (not np.isnan(val))
            elif attribute == 'nonempty':
                res = (hasattr(val, 'size') and val.size > 0) or (not hasattr(val, 'size') and val is not None)
            elif attribute == 'diag':
                res = np.all(val == np.diag(np.diag(val)))
            elif attribute == 'upper':
                res = np.all(np.triu(val) == val)
            elif attribute == 'lower':
                res = np.all(np.tril(val) == val)
            elif attribute == 'full':
                res = np.all(np.isfinite(val))
            elif attribute == 'sparse':
                # This would typically involve checking the type or properties of a sparse matrix object
                res = False # Placeholder, needs actual sparse matrix check
            elif attribute == 'logical':
                res = np.issubdtype(val.dtype, np.bool_) if isinstance(val, np.ndarray) else isinstance(val, bool)
            elif attribute == 'sym':
                res = np.allclose(val, val.T)
            elif attribute == 'skew':
                res = np.allclose(val, -val.T)
            elif attribute == 'hermitian':
                res = np.allclose(val, np.conjugate(val.T))
            elif attribute == 'positiveSemidefinite':
                res = np.all(np.linalg.eigvals(val) >= 0)
            elif attribute == 'positiveDefinite':
                res = np.all(np.linalg.eigvals(val) > 0)
            elif attribute == 'nonnegative':
                res = np.all(val >= 0)
            elif attribute == 'column':
                res = (isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == 1)
            elif attribute == 'vector':
                res = (isinstance(val, np.ndarray) and (val.ndim == 1 or (val.ndim == 2 and (val.shape[0] == 1 or val.shape[1] == 1))))
            elif attribute == 'matrix':
                res = (isinstance(val, np.ndarray) and val.ndim == 2)
            else:
                # Attempt to evaluate as a built-in function or class check (e.g., 'isinstance')
                try:
                    if attribute.startswith('is'):
                        func_name = attribute
                    else:
                        func_name = 'is' + attribute[0].upper() + attribute[1:]
                    
                    # Special handling for `isnumeric` for numpy arrays
                    if func_name == 'isnumeric':
                        res = np.issubdtype(val.dtype, np.number) if isinstance(val, np.ndarray) else isinstance(val, (int, float, complex))
                    elif func_name == 'isequal': # assuming this is comparison to another value
                        res = np.array_equal(val, kwargs.get('other_val', None))
                    elif func_name == 'islogical':
                        res = np.issubdtype(val.dtype, np.bool_) if isinstance(val, np.ndarray) else isinstance(val, bool)
                    elif func_name == 'ischar':
                        res = isinstance(val, str)
                    elif func_name == 'iscell':
                        res = isinstance(val, list)
                    elif func_name == 'istable': # No direct Python equivalent, often implies pandas DataFrame
                        res = False # Placeholder
                    else:
                        # Try to find a global function or a method on value
                        if hasattr(np, func_name):
                            res = getattr(np, func_name)(val)
                        elif hasattr(val, func_name):
                            res = getattr(val, func_name)()
                        else:
                            raise CORAerror('CORA:wrongValue', 'third', f"Unknown attribute or function: {attribute}")
                except Exception as e:
                    raise CORAerror('CORA:wrongValue', 'third', f"Unable to check attribute {attribute} for {val}: {e}")

        elif isinstance(attribute, Callable):
            # evaluate function handle
            res = attribute(val)
        else:
            # unable to check attribute; unknown type
            raise CORAerror('CORA:wrongValue', 'third', f"Unable to check attribute {attribute} for {val}")

        # apply reduction method
        if reduction == 'all':
            if isinstance(res, np.ndarray):
                res = np.all(res)
        elif reduction == 'any':
            if isinstance(res, np.ndarray):
                res = np.any(res)
        elif reduction == 'none':
            pass  # res = res;
        else:
            raise CORAerror('CORA:wrongValue', 'fourth', "{'all','any','none'}")
        
        return bool(res)

    # init
    resvec = [False] * (len(attributes) + 1)

    # check class
    # For Python, `isinstance` is used. Class names like 'numeric' need mapping.
    class_check_passed = False
    if not class_name:
        class_check_passed = True
    elif class_name == 'numeric':
        class_check_passed = isinstance(value, (int, float, np.number)) or (isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number))
    elif class_name == 'char' or class_name == 'string':
        class_check_passed = isinstance(value, str)
    elif class_name == 'logical':
        class_check_passed = isinstance(value, bool) or (isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.bool_))
    elif class_name == 'cell':
        class_check_passed = isinstance(value, list)
    elif class_name == 'struct':
        class_check_passed = isinstance(value, dict)
    elif class_name == 'function_handle':
        class_check_passed = isinstance(value, Callable)
    elif class_name == 'numpy.ndarray':
        class_check_passed = isinstance(value, np.ndarray)
    else:
        try:
            # Try to evaluate class_name as a Python class
            # This requires the class to be imported in the scope where this function is called
            # For now, a simplified approach. A more robust solution might use a class registry.
            target_class = globals().get(class_name, None)
            if target_class is None:
                # If not found globally, try to import dynamically (less ideal but might be needed)
                # This is complex and potentially unsafe for a general-purpose validator.
                # For now, let's assume it's a known CORA class that should be imported.
                pass # The outer module needs to import it.
            class_check_passed = isinstance(value, target_class) if target_class else False
        except Exception:
            class_check_passed = False

    resvec[0] = class_check_passed

    # check attributes
    for i in range(len(attributes)):
        if not resvec[i]: # If previous check failed, no need to continue
            break
        resvec[i+1] = aux_checkAttribute(value, class_name, attributes[i], 'all')
    
    # gather results
    res = all(resvec)
    return res 