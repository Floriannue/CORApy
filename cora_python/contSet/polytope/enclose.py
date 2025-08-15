"""
enclose - computes a polytope that encloses a polytope and its linear transformation

Exact MATLAB translation: if given P2 or (M,Pplus), return convHull(P1,P2).
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def enclose(P1, *varargin):
    from .polytope import Polytope

    if not isinstance(P1, Polytope):
        raise CORAerror('CORA:wrongInput', 'First argument must be a polytope')

    if len(varargin) == 1:
        P2 = varargin[0]
        if not isinstance(P2, Polytope):
            raise CORAerror('CORA:wrongInput', 'Second argument must be a polytope when only two inputs are given')
    elif len(varargin) == 2:
        M, Pplus = varargin
        P2 = (M @ P1) + Pplus
    else:
        raise CORAerror('CORA:wrongInput', 'Wrong number of input arguments')

    return P1.convHull_(P2)


