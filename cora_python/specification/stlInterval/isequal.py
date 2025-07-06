def isequal(obj, other, tol=None):
    """
    isequal - checks if two stlInterval objects are equal
    
    Syntax:
        res = isequal(obj, other)
        res = isequal(obj, other, tol)
    
    Inputs:
        obj - stlInterval object
        other - stlInterval object
        tol - (optional) tolerance (default: eps)
    
    Outputs:
        res - True if the stlIntervals are equal, False otherwise
    
    Example:
        I1 = StlInterval(1, 2, True, True)
        I2 = StlInterval(1, 2, False, True)
        res = isequal(I1, I2)  # False
    
    Authors:       Florian Lercher (MATLAB)
                   Python translation by AI Assistant
    Written:       06-February-2024 (MATLAB)
    Python translation: 2025
    """
    
    from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
    from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
    import numpy as np
    
    # Check number of arguments (2-3)
    if tol is None:
        tol = np.finfo(float).eps
    
    # Input validation
    inputArgsCheck([
        [obj, 'att', 'stlInterval'],
        [other, 'att', 'stlInterval'], 
        [tol, 'att', 'numeric', ['nonnan', 'scalar', 'nonnegative']]
    ])
    
    # Check if either is empty
    if obj.isemptyobject() or other.isemptyobject():
        return obj.isemptyobject() and other.isemptyobject()
    else:
        # Compare bounds with tolerance and closure properties
        return (withinTol(obj.lower, other.lower, tol) and 
                withinTol(obj.upper, other.upper, tol) and
                obj.leftClosed == other.leftClosed and 
                obj.rightClosed == other.rightClosed) 