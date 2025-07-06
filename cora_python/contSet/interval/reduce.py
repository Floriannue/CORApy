def reduce(I, *varargin):
    """
    reduce - reduces the set representation size of an interval; this
       function exists only for operator overloading purposes as the set
       representation size of an interval object is static

    Syntax:
       I = reduce(I)

    Inputs:
       I - interval object

    Outputs:
       I - interval object
    """
    # no content, returns the object itself
    return I 