from .interval import Interval

def convHull_(I1, I2, *varargin):
    """
    convHull_ - computes the convex hull of an interval and another set or a
       point. For intervals, this is equivalent to the union.
    
    Syntax:
       res = convHull_(I,S)
    
    Inputs:
       I1 - interval object
       I2 - interval object, or numeric value
    
    Outputs:
       S_out - convex hull of I1 and I2
    """
    
    from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric
    from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
    
    # ensure that numeric is second input argument
    I1, I2 = reorder_numeric(I1, I2)
    
    # convert numeric to interval if needed
    if not isinstance(I2, Interval):
        I2 = Interval(I2)
        
    # check dimensions
    equal_dim_check(I1, I2)
    
    # empty set cases
    if I1.is_empty():
        return I2
    if I2.is_empty():
        return I1
    
    # call or function
    return I1 | I2 