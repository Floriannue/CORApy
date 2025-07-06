import numpy as np
from typing import Union

from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def mrdivide(numerator: Interval, denominator: Union[Interval, float]):
    """
    mrdivide - Overloaded matrix division '/' operator for intervals
    This function is only defined for a scalar denominator.

    Syntax:
        res = mrdivide(numerator, denominator)

    Inputs:
        numerator - interval object
        denominator - interval object or scalar

    Outputs:
        res - interval object
    """

    is_scalar_denominator = False
    if isinstance(denominator, Interval):
        if denominator.isscalar():
            is_scalar_denominator = True
    elif np.isscalar(denominator):
        is_scalar_denominator = True

    if is_scalar_denominator:
        # Defer to element-wise division, which is overloaded as '/'
        return numerator / denominator
    else:
        raise CORAerror('CORA:noops', numerator, denominator) 