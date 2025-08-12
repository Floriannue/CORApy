import numpy as np
from typing import Any, Callable, List, Union
import logging

# Do not set global logging config here
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def checkValueAttributes(value: Any, check_type: str, attributes) -> bool:
    logger.debug("checkValueAttributes: value=%s, check_type=%s, attributes=%s", value, check_type, attributes)

    """
    checkValueAttributes - checks if the given value is of the correct class
    and has the correct attributes
    """

    # Normalize inputs
    class_name = check_type if isinstance(check_type, str) else ''
    if attributes is None:
        attributes = []
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

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
        val = np.asarray(val) if isinstance(val, list) else val
        return np.all(val == np.floor(val))

    def aux_iseven(val: Any) -> bool:
        val = np.asarray(val) if isinstance(val, list) else val
        return np.all(val % 2 == 0)

    def aux_isodd(val: Any) -> bool:
        val = np.asarray(val) if isinstance(val, list) else val
        return np.all(val % 2 != 0)

    def aux_ispositive(val: Any) -> bool:
        val = np.asarray(val) if isinstance(val, list) else val
        return np.all(val > 0)

    def aux_isnegative(val: Any) -> bool:
        val = np.asarray(val) if isinstance(val, list) else val
        return np.all(val < 0)

    def aux_iszero(val: Any) -> bool:
        val = np.asarray(val) if isinstance(val, list) else val
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
                if class_name.lower() in ['ellipsoid', 'interval', 'zonotope', 'polytope', 'contset']:
                    res = not isinstance(value, (list, np.ndarray)) or (isinstance(value, np.ndarray) and value.shape == ())
                elif class_name.lower() == 'numeric':
                    res = np.isscalar(value) or \
                          (isinstance(value, np.ndarray) and \
                           (value.size == 1 or \
                            value.ndim == 1 or \
                            (value.ndim == 2 and (value.shape[0] == 1 or value.shape[1] == 1)) \
                           ))
                else:
                    res = np.isscalar(value)
            elif attribute == 'row':
                res = (isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] == 1)
            elif attribute == 'column':
                res = (isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == 1)
            elif attribute == 'vector':
                is_list_of_nums = isinstance(val, list) and all(isinstance(x, (int, float, np.number)) for x in val)
                is_numpy_vector = isinstance(val, np.ndarray) and (val.ndim == 1 or (val.ndim == 2 and (val.shape[0] == 1 or val.shape[1] == 1)))
                res = is_list_of_nums or is_numpy_vector
            elif attribute == 'real':
                res = np.isreal(val).all()
            elif attribute == 'finite':
                res = np.isfinite(val).all()
            elif attribute == 'nonnan':
                res = (not np.isnan(val).any()) if isinstance(val, np.ndarray) else (not np.isnan(val))
            elif attribute == 'nan':
                res = np.isnan(val).any() if isinstance(val, np.ndarray) else np.isnan(val)
            elif attribute == 'empty':
                if val is None:
                    res = True
                elif hasattr(val, 'size'):
                    res = val.size == 0
                elif hasattr(val, '__len__'):
                    res = len(val) == 0
                else:
                    res = False
            elif attribute == 'nonempty':
                if val is None:
                    res = False
                elif hasattr(val, 'size'):
                    res = val.size > 0
                elif hasattr(val, '__len__'):
                    res = len(val) > 0
                else:
                    res = True
            elif attribute == 'diag':
                res = np.all(val == np.diag(np.diag(val)))
            elif attribute == 'upper':
                res = np.all(np.triu(val) == val)
            elif attribute == 'lower':
                res = np.all(np.tril(val) == val)
            elif attribute == 'full':
                res = np.all(np.isfinite(val))
            elif attribute == 'sparse':
                res = False
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
            elif attribute == 'matrix':
                res = (isinstance(val, np.ndarray) and val.ndim == 2)
            else:
                try:
                    if attribute.startswith('is'):
                        func_name = attribute
                    else:
                        func_name = 'is' + attribute[0].upper() + attribute[1:]
                    if func_name == 'isnumeric':
                        res = np.issubdtype(val.dtype, np.number) if isinstance(val, np.ndarray) else isinstance(val, (int, float, complex))
                    elif func_name == 'isequal':
                        res = np.array_equal(val, kwargs.get('other_val', None))
                    elif func_name == 'islogical':
                        res = np.issubdtype(val.dtype, np.bool_) if isinstance(val, np.ndarray) else isinstance(val, bool)
                    elif func_name == 'ischar':
                        res = isinstance(val, str)
                    elif func_name == 'iscell':
                        res = isinstance(val, list)
                    elif func_name == 'istable':
                        res = False
                    else:
                        if hasattr(np, func_name):
                            res = getattr(np, func_name)(val)
                        elif hasattr(val, func_name):
                            res = getattr(val, func_name)()
                        else:
                            raise CORAerror('CORA:wrongValue', 'third', f"Unknown attribute or function: {attribute}")
                except Exception as e:
                    raise CORAerror('CORA:wrongValue', 'third', f"Unable to check attribute {attribute} for {val}: {e}")

        elif isinstance(attribute, Callable):
            res = attribute(val)
        else:
            raise CORAerror('CORA:wrongValue', 'third', f"Unable to check attribute {attribute} for {val}")

        if reduction == 'all':
            if isinstance(res, np.ndarray):
                res = np.all(res)
        elif reduction == 'any':
            if isinstance(res, np.ndarray):
                res = np.any(res)
        elif reduction == 'none':
            pass
        else:
            raise CORAerror('CORA:wrongValue', 'fourth', "{'all','any','none'}")
        
        return bool(res)

    # init
    resvec = [False] * (len(attributes) + 1)

    # check class
    class_check_passed = False
    if not class_name:
        class_check_passed = True
    elif class_name == 'numeric':
        class_check_passed = (isinstance(value, (int, float, np.number)) or 
                             (isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)) or
                             (isinstance(value, list) and all(isinstance(x, (int, float, np.number)) for x in value)))
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
            mro = type(value).mro()
            class_check_passed = any(c.__name__.lower() == class_name.lower() for c in mro)
        except Exception:
            class_check_passed = False

    resvec[0] = class_check_passed

    # check attributes
    for i in range(len(attributes)):
        if not resvec[i]:
            break
        resvec[i+1] = aux_checkAttribute(value, class_name, attributes[i], 'all')
    
    res = all(resvec)
    return res 