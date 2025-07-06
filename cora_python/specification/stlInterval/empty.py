def empty(n=None):
    """
    empty - creates an empty stlInterval
    
    Syntax:
        obj = empty()
        obj = empty(n)
    
    Inputs:
        n - dimension (must be 1 for stlInterval)
    
    Outputs:
        obj - empty stlInterval object
    
    Authors:       Florian Lercher (MATLAB)
                   Python translation by AI Assistant
    Written:       06-February-2024 (MATLAB)
    Python translation: 2025
    """
    
    from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    from .stlInterval import StlInterval
    
    # Parse input
    if n is None:
        n = 1
    
    # Validate input
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative']]])
    
    if n != 1:
        raise CORAerror('CORA:wrongValue', 'first', 'STL intervals are always one-dimensional.')
    
    # Create empty stlInterval (no arguments creates empty interval)
    return StlInterval() 