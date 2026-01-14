import numpy as np
from typing import Any, Callable, List, Union, Dict, Type
import logging
import sympy as sp
from scipy import sparse

# Do not set global logging config here
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Map MATLAB class names to Python classes (lazy import to avoid circular dependencies)
_CLASS_MAP: Dict[str, Type] = {}

def _get_class_for_name(class_name: str) -> Type:
    """Get Python class for MATLAB class name (lazy import)"""
    if class_name in _CLASS_MAP:
        return _CLASS_MAP[class_name]
    
    # Import classes on first use
    if class_name == 'abstractReset':
        from cora_python.hybridDynamics.abstractReset.abstractReset import AbstractReset
        _CLASS_MAP[class_name] = AbstractReset
        return AbstractReset
    elif class_name == 'linearReset':
        from cora_python.hybridDynamics.linearReset.linearReset import LinearReset
        _CLASS_MAP[class_name] = LinearReset
        return LinearReset
    elif class_name == 'nonlinearReset':
        from cora_python.hybridDynamics.nonlinearReset.nonlinearReset import NonlinearReset
        _CLASS_MAP[class_name] = NonlinearReset
        return NonlinearReset
    elif class_name == 'transition':
        from cora_python.hybridDynamics.transition.transition import Transition
        _CLASS_MAP[class_name] = Transition
        return Transition
    elif class_name == 'location':
        from cora_python.hybridDynamics.location.location import Location
        _CLASS_MAP[class_name] = Location
        return Location
    elif class_name == 'hybridAutomaton':
        from cora_python.hybridDynamics.hybridAutomaton.hybridAutomaton import HybridAutomaton
        _CLASS_MAP[class_name] = HybridAutomaton
        return HybridAutomaton
    elif class_name == 'specification':
        from cora_python.specification.specification.specification import Specification
        _CLASS_MAP[class_name] = Specification
        return Specification
    elif class_name == 'stl':
        # STL class might not exist yet, but we can try to import it
        try:
            from cora_python.specification.stl.stl import Stl
            _CLASS_MAP[class_name] = Stl
            return Stl
        except ImportError:
            pass
    
    return None

def checkValueAttributes(value: Any, check_type: str, attributes) -> bool:
    logger.debug("checkValueAttributes: value=%s, check_type=%s, attributes=%s", value, check_type, attributes)

    """
    checkValueAttributes - checks if the given value is of the correct class
    and has the correct attributes
    """

    # Normalize inputs
    # Handle tuple of class names (e.g., ('abstractReset', 'struct'))
    if isinstance(check_type, tuple):
        # Check if value matches any of the classes in the tuple
        for cls_name in check_type:
            if checkValueAttributes(value, cls_name, attributes):
                return True
        return False
    
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
                elif class_name.lower() == 'function_handle':
                    # A function handle is always "scalar" (single function, not array of functions)
                    res = isinstance(value, Callable)
                else:
                    # For object classes (e.g., abstractReset, linearReset, etc.), scalar means single instance, not array
                    # MATLAB: scalar for objects means not an array of objects
                    res = not isinstance(value, (list, tuple, np.ndarray)) or \
                          (isinstance(value, np.ndarray) and value.size == 0)
            elif attribute == 'row':
                res = (isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] == 1)
            elif attribute == 'column':
                # MATLAB: column means column vector or scalar (1x1 is considered column)
                # Scalar numeric values are also acceptable as "column" (they can be reshaped)
                if isinstance(val, (int, float, np.number)):
                    res = True  # Scalar is acceptable as column
                elif isinstance(val, np.ndarray):
                    res = (val.ndim == 2 and val.shape[1] == 1) or (val.ndim == 0) or (val.size == 1)
                else:
                    res = False
            elif attribute == 'vector':
                is_list_of_nums = isinstance(val, list) and all(isinstance(x, (int, float, np.number)) for x in val)
                is_numpy_vector = isinstance(val, np.ndarray) and (val.ndim == 1 or (val.ndim == 2 and (val.shape[0] == 1 or val.shape[1] == 1)))
                res = is_list_of_nums or is_numpy_vector
            elif attribute == 'real':
                res = np.isreal(val).all()
            elif attribute == 'finite':
                # Empty arrays are considered finite
                if hasattr(val, 'size') and val.size == 0:
                    res = True
                else:
                    res = np.isfinite(val).all()
            elif attribute == 'nonnan':
                # Handle sparse matrices
                from scipy import sparse
                if sparse.issparse(val):
                    res = not np.isnan(val.data).any() if val.data.size > 0 else True
                elif isinstance(val, np.ndarray):
                    res = (not np.isnan(val).any())
                else:
                    res = (not np.isnan(val))
            elif attribute == 'nan':
                # Handle sparse matrices
                from scipy import sparse
                if sparse.issparse(val):
                    res = np.isnan(val.data).any() if val.data.size > 0 else False
                elif isinstance(val, np.ndarray):
                    res = np.isnan(val).any()
                else:
                    res = np.isnan(val)
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
        # Check for sparse matrices (scipy.sparse)
        is_sparse_numeric = sparse.issparse(value) and np.issubdtype(value.dtype, np.number)
        class_check_passed = (isinstance(value, (int, float, np.number)) or 
                             (isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)) or
                             is_sparse_numeric or
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
    elif class_name == 'sym':
        # Check if value is a sympy object (Basic is the base class for all sympy objects)
        class_check_passed = isinstance(value, sp.Basic) or isinstance(value, sp.Matrix)
    elif class_name == 'numpy.ndarray':
        class_check_passed = isinstance(value, np.ndarray)
    else:
        # MATLAB: isa(value, class) - check if value is instance of class
        # Handle lists of objects (e.g., list of specifications)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            # Check if all items in the list are of the correct class
            python_class = _get_class_for_name(class_name)
            if python_class is not None:
                class_check_passed = all(isinstance(item, python_class) for item in value)
            else:
                # Fallback: check MRO for class name match for all items
                class_check_passed = all(
                    any(c.__name__.lower() == class_name.lower() for c in type(item).mro())
                    for item in value
                )
        else:
            # Try to get the Python class for the MATLAB class name
            python_class = _get_class_for_name(class_name)
            if python_class is not None:
                # Direct isinstance check (matches MATLAB's isa)
                class_check_passed = isinstance(value, python_class)
            else:
                # Fallback: check MRO for class name match
                mro = type(value).mro()
                class_check_passed = any(c.__name__.lower() == class_name.lower() for c in mro)

    resvec[0] = class_check_passed

    # check attributes
    for i in range(len(attributes)):
        if not resvec[i]:
            break
        resvec[i+1] = aux_checkAttribute(value, class_name, attributes[i], 'all')
    
    res = all(resvec)
    return res 