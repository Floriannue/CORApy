import numpy as np

def ctranspose(I: 'Interval') -> 'Interval':
    """
    ctranspose - Overloaded ''' operator for single operand
    For real intervals, this is the same as transpose.

    Syntax:
        res = ctranspose(I)

    Inputs:
        I - interval object

    Outputs:
        res - interval object
    """
    
    # Create a new interval object for the result
    res = I.copy()
    
    # Transpose inf and sup
    res.inf = I.inf.T
    res.sup = I.sup.T
    
    return res 