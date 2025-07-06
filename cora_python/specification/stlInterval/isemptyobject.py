def isemptyobject(obj):
    """
    isemptyobject - checks if a stlInterval object is empty
    
    Syntax:
        res = isemptyobject(obj)
    
    Inputs:
        obj - stlInterval object
    
    Outputs:
        res - True if the stlInterval is empty, False otherwise
    
    Example:
        I = StlInterval(1, 2)
        isemptyobject(I)  # False
    
    Authors:       Florian Lercher (MATLAB)
                   Python translation by AI Assistant
    Written:       06-February-2024 (MATLAB)
    Python translation: 2025
    """
    
    # stlInterval is empty if lower bound is None (equivalent to MATLAB's isempty)
    return obj.lower is None 