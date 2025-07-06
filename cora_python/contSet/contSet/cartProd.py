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

    # In the MATLAB version, some basic parsing happens here, 
    # but the core logic and type checking is delegated to the 
    # subclass-specific cartProd_ method.
    
    try:
        # Check which argument is the contSet instance and call the method on it
        if isinstance(S1, ContSet):
            res = S1.cartProd_(S2, *varargin)
        elif isinstance(S2, ContSet):
            # If S1 is numeric, S2 must implement the logic
            res = S2.cartProd_(S1, *varargin)
        else:
            # This case should ideally not be reached if called from an instance
            raise CORAerror('CORA:noops', S1, S2)

    except Exception as ME:
        # The MATLAB code has a specific check for empty sets here.
        # Replicating that behavior.
        if hasattr(S1, 'is_empty') and hasattr(S2, 'is_empty'):
             if S1.is_empty() or S2.is_empty():
                raise CORAerror('CORA:notSupported', "Cartesian products with empty sets are not supported.")
        
        # If it's not an empty set issue, rethrow the original error.
        raise ME
            
    return res 