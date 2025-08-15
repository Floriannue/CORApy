import numpy as np
from typing import Union

from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck

def mtimes(factor1: Union[np.ndarray, float, int, PolyZonotope], factor2: Union[np.ndarray, float, int, PolyZonotope]) -> PolyZonotope:
    """
    mtimes - Overloaded '*' operator for the multiplication of a matrix or an
    interval matrix with a polynomial zonotope

    Syntax:
        pZ = factor1 * factor2
        pZ = mtimes(factor1,factor2)

    Inputs:
        factor1 - polyZonotope object, numeric matrix or scalar
        factor2 - polyZonotope object, numeric scalar

    Outputs:
        pZ - polyZonotope after the multiplication
    """
    try:
        # numeric matrix/scalar * polynomial zonotope
        if isinstance(factor1, (np.ndarray, float, int)) and isinstance(factor2, PolyZonotope):
            c = factor1 * factor2.c
            G = factor1 * factor2.G
            GI = factor1 * factor2.GI
            pZ = PolyZonotope(c, G, GI, factor2.E, factor2.id)
            return pZ

        # polynomial zonotope * scalar
        # (note that polynomial zonotope * matrix is not supported)
        if isinstance(factor2, (float, int)) and isinstance(factor1, PolyZonotope):
            c = factor2 * factor1.c
            G = factor2 * factor1.G
            GI = factor2 * factor1.GI
            pZ = PolyZonotope(c, G, GI, factor1.E, factor1.id)
            return pZ

    except Exception as ME:
        # check whether different dimension of ambient space
        # This equalDimCheck will raise an error if dimensions are not equal,
        # so rethrow only if equalDimCheck doesn't catch it or is not applicable.
        try:
            equalDimCheck(factor1, factor2)
        except CORAerror:
            # If equalDimCheck itself raised a CORAerror, re-raise it.
            raise
        
        # If equalDimCheck did not raise an error, but another error occurred,
        # re-raise the original exception.
        raise ME

    raise CORAerror('CORA:noops', factor1, factor2)
