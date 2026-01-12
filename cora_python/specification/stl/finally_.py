"""
finally_ - finally-operator for Signal Temporal Logic

TRANSLATED FROM: cora_matlab/specification/@stl/finally.m

Note: Named 'finally_' because 'finally' is a reserved keyword in Python

Syntax:
    res = finally_(obj, time)

Inputs:
    obj - logic formula (class stl)
    time - time interval (class interval)

Outputs:
    res - resulting stl formula (class stl)

Example: 
    x = stl('x',2);
    eq = finally(x(1) < 5,interval(0.1,0.2))

Authors:       Niklas Kochdumper, Benedikt Seidl
Written:       09-November-2022
Last update:   07-February-2024 (FL, replace from and to by interval)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .stl import Stl

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.specification.stlInterval.stlInterval import StlInterval
from cora_python.contSet.interval import Interval


def finally_(obj: 'Stl', time: Any) -> 'Stl':
    """
    Finally-operator for Signal Temporal Logic
    
    Args:
        obj: logic formula (class stl)
        time: time interval (class interval or stlInterval)
    
    Returns:
        Stl: resulting stl formula
    """
    # Check input arguments
    if not obj.logic:
        raise CORAerror('CORA:notSupported',
                      'This operation is not supported for stl objects!')
    
    # Convert time to stlInterval if needed
    if isinstance(time, Interval):
        if time.dim() != 1:
            raise CORAerror('CORA:wrongValue',
                          'Wrong format for input argument "time"!')
        time = StlInterval(time)
    elif not isinstance(time, StlInterval):
        raise CORAerror('CORA:wrongValue',
                      'Wrong format for input argument "time"!')
    
    # Construct resulting stl object
    from .stl import Stl
    res = Stl.__new__(Stl)  # Create new instance without calling __init__
    res.type = 'finally'
    res.lhs = obj
    res.rhs = None
    res.id = None
    res.temporal = True
    res.interval = time
    res.variables = obj.variables.copy() if hasattr(obj, 'variables') else []
    res.var = getattr(obj, 'var', None)
    res.logic = True
    return res

