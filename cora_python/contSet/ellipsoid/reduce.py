"""
Authors: Mark Wetzlinger
Written: 17-March-2023
Last update: ---
Last revision: ---

Automatic python translation: Florian Nüssel BA 2025
"""
def reduce(self, *args, **kwargs):
    """
    reduce - reduces the set representation size of an ellipsoid; this
    function exists only for operator overloading purposes as the set
    representation size of an ellipsoid object is static

    Syntax:
        E = reduce(E)

    Inputs:
        E - ellipsoid object

    Outputs:
        E - ellipsoid object

    Example:
        E = Ellipsoid([[1,0.5],[0.5,3]],[[1],[-1]])
        E_ = reduce(E)
    """

    # no content...
    return self