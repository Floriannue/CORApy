from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def stlInterval(I):
    """
    Converts an interval object to an stlInterval object.
    """
    from cora_python.specification.stlInterval import StlInterval
    # Interval must be 1D
    if I.dim() != 1:
        raise CORAerror('CORA:wrongValue', 'Interval must be one-dimensional.')
        
    # Regular intervals are always closed
    return StlInterval(I.inf, I.sup, True, True) 