
def intervalMatrix(I):
    """
    Converts an interval object to an intervalMatrix object.
    """
    from cora_python.matrixSet.intervalMatrix.intervalMatrix import IntervalMatrix
    # Construct an IntervalMatrix from the center and radius of the interval
    return IntervalMatrix(I.center(), I.rad()) 