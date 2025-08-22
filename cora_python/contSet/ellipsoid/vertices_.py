"""
vertices_ - computes the vertices of an ellipsoid (only 1D supported)

Syntax:
    V = vertices_(E)

Inputs:
    E - ellipsoid object

Outputs:
    V - vertices

Example: 
    E = Ellipsoid(1, 1)
    V = vertices_(E)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/vertices

Authors:       Mark Wetzlinger (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       25-July-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

def vertices_(E, method: str = 'convHull', *args, **kwargs):
    """
    Computes the vertices of a 1D ellipsoid.
    Args:
        E: ellipsoid object
        method: Method for vertex computation (unused, kept for interface compatibility)
        *args: Additional arguments (unused)
        **kwargs: Additional keyword arguments (unused)
    Returns:
        V: vertices (np.ndarray)
    """
    n = E.dim()
    if E.isemptyobject():
        return np.empty((n, 0))
    if n > 1:
        raise CORAerror('CORA:notSupported', 'Vertex computation of ellipsoids only supported for 1D cases.')

    # Compute minimum and maximum using support function
    _, Vmin = E.supportFunc_(-np.ones((1, 1)), 'upper')
    _, Vmax = E.supportFunc_(np.ones((1, 1)), 'upper')

    if withinTol(Vmin, Vmax):
        # Only return one vertex
        V = Vmin
    else:
        V = np.hstack((Vmin, Vmax))
    return V 