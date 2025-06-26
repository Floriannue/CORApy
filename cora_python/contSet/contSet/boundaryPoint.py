import numpy as np

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing import setDefaultValues
from cora_python.g.functions.matlab.validate.check import withinTol, equal_dim_check

def boundaryPoint(S, dir, *varargin):
    """
    computes the point on the boundary of a set along a
    given direction, starting from a given start point, or, by default,
    from the center of the set; for unbounded sets, a start point must be
    provided; any given start point must be contained in the set; note
    that the vector may immediately reach the boundary of degenerate sets

    Syntax:
        x = boundaryPoint(S,dir)
        x = boundaryPoint(S,dir,startPoint)

    Inputs:
        S - contSet object
        dir - direction
        startPoint - start point for the direction vector

    Outputs:
        x - point on the boundary of the zonotope
    """

    # default starting point is the center: caution, as this may be
    # time-consuming to compute
    startPoint = setDefaultValues([S.center()], varargin)[0]
    if np.any(np.isnan(startPoint)):
        raise CORAerror("CORA:wrongValue", 'third', 'For unbounded sets, a start point must be provided')

    if np.all(withinTol(dir, 0)):
        raise CORAerror("CORA:wrongValue", 'second', 'Vector has to be non-zero.')

    # check for dimensions
    equal_dim_check(S, dir)
    equal_dim_check(S, startPoint)

    # for empty sets, return empty
    if S.representsa_('emptySet', 0):
        return np.zeros((S.dim(), 0))

    # start point must be contained in the set
    if not S.contains(startPoint):
        raise CORAerror('CORA:wrongValue', 'third', 'Start point must be contained in the set.')

    # not implemented in subclasses
    raise CORAerror('CORA:noops', S) 