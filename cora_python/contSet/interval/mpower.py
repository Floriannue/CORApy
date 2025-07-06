"""
mpower - Overloaded '^' operator for intervals (matrix power)

Syntax:
    res = mpower(base, exponent)

Inputs:
    base - interval object or numerical value
    exponent - interval object or numerical value

Outputs:
    res - interval object

Example:
    base = Interval([[-2], [1]], [[3], [2]])
    exponent = 2
    base.mpower(exponent)

Authors: Matthias Althoff, Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 25-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np

def mpower(base, exponent):
    """
    Overloaded matrix power operator for intervals
    
    Args:
        base: Interval object or numerical value
        exponent: Interval object or numerical value (must be scalar for matrix case)
        
    Returns:
        Interval object result of matrix power operation
    """
    from .interval import Interval
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # Scalar case - delegate to element-wise power
    if isinstance(base, Interval) and base.inf.size == 1:
        if np.isscalar(exponent):
            # Use element-wise power for scalars (allows negative/fractional exponents)
            return base.power(exponent)
        else:
            raise CORAerror('CORA:wrongValue', 'second', 'scalar exponent for scalar base')
    
    # Matrix case
    elif isinstance(base, Interval):
        if np.isscalar(exponent) and exponent > 0:
            # Check if exponent is an integer
            if not (isinstance(exponent, (int, np.integer)) or 
                    (isinstance(exponent, (float, np.floating)) and exponent == int(exponent))):
                raise CORAerror('CORA:wrongValue', 'second', 'integer exponent for matrix power')
            
            # Integer matrix power
            exponent = int(exponent)
            res = base
            for i in range(2, exponent + 1):
                res = res @ base  # Use matrix multiplication
            return res
            
        elif np.isscalar(exponent) and exponent == 0:
            # Check if base is square matrix
            if base.inf.shape[0] == base.inf.shape[1]:
                # Return identity matrix
                n = base.inf.shape[0]
                identity = np.eye(n)
                return Interval(identity, identity)
            else:
                raise CORAerror('CORA:wrongValue', 'first', 'square matrix for exponent 0')
                
        else:
            raise CORAerror('CORA:wrongValue', 'second', 'square matrix')
    
    else:
        # Numeric case - use numpy power
        return np.power(base, exponent) 