"""
lift_ - lifts an ellipsoid onto a higher-dimensional space

Syntax:
   E = lift_(E,N,proj)

Inputs:
   E - ellipsoid object
   N - dimension of the higher-dimensional space
   proj - states of the high-dimensional space that correspond to the
         states of the low-dimensional space

Outputs:
   E - projected ellipsoid

Authors:       Tobias Ladner
Written:       19-September-2023
Automatic python translation: Florian Nüssel BA 2025
"""

from __future__ import annotations

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def lift_(E: Ellipsoid, N: int, proj):
    if len(proj) == N:
        # use project
        return E.project(proj)
    else:
        # projection to higher dimension is not defined as function expects new
        # dimensions to be unbounded
        raise CORAerror('CORA:notDefined',
                        'New dimensions cannot be unbounded as the set representation is always bounded.',
                        'contSet/projectHighDim')

