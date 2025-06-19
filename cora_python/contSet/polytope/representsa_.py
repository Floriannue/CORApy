from typing import Tuple, Union
import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.check import withinTol

def representsa_(p: Polytope, set_type: str, tol: float = 1e-9, **kwargs) -> Union[bool, Tuple[bool, Polytope]]:
    """
    Checks if a polytope can be represented by another set type.
    """
    res = False
    p_conv = None

    if set_type == 'emptySet':
        res = p.is_empty()
    elif set_type == 'point':
        if p.is_empty():
            res = False
        else:
            # check if all vertices are the same
            V = p.V
            res = np.all(withinTol(V, V[:, [0]], tol))
    elif set_type == 'fullspace':
        res = not p.is_bounded()
    elif set_type == 'origin':
        if p.is_empty():
            res = False
        else:
            V = p.V
            res = np.all(withinTol(V, 0, tol))

    if 'return_set' in kwargs and kwargs['return_set']:
        if res:
            # Note: Conversion logic to other set types would go here.
            # For now, if the representation is valid, we can consider
            # creating a new object of the target type if needed,
            # but for basic checks, returning the original object is sufficient.
            p_conv = p
        return res, p_conv
    else:
        return res 