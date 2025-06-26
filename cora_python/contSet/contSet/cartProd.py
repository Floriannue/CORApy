import numpy as np

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.contSet import ContSet


def cartProd(S1, S2, *varargin):
    """
    computes the Cartesian product of two sets

    Description:
        computes the set { [s_1 s_2 ]^T | s_1 in S_1, s_2 in S_2 }.

    Syntax:
        res = cartProd(S1,S2)
        res = cartProd(S1,S2,type)

    Inputs:
        S1,S2 - contSet object
        type - type of computation ('exact','inner','outer')

    Outputs:
        res - Cartesian product
    """

    # parse input arguments
    if isinstance(S1, Ellipsoid) or isinstance(S2, Ellipsoid):
        type = setDefaultValues(['outer'], varargin)[0]
    else:
        type = setDefaultValues(['exact'], varargin)[0]

    # check input arguments: two versions as order of input arguments matters
    try:
        inputArgsCheck([[S1, 'att', ['contSet', 'numeric']],
                        [S2, 'att', ['contSet', 'numeric'], 'vector'],
                        [type, 'str', ['outer', 'inner', 'exact']]])
    except Exception:
        inputArgsCheck([[S1, 'att', ['contSet', 'numeric'], 'vector'],
                        [S2, 'att', ['contSet', 'numeric']],
                        [type, 'str', ['outer', 'inner', 'exact']]])

    # call subclass method
    try:
        # Since cartProd is a method of S1, we call it on S1
        res = S1.cartProd_(S2, type)
    except Exception as ME:
        # Cartesian products with empty sets are currently not supported,
        # because we cannot concatenate empty vectors with filled vectors
        if S1.representsa_('emptySet', 1e-8) or S2.representsa_('emptySet', 1e-8):
            raise CORAerror("CORA:notSupported", "Cartesian products with empty sets are not supported.")
        else:
            raise ME
            
    return res 