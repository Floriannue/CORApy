import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

def mtimes(factor1, factor2) -> 'Ellipsoid':
    """
    mtimes - Overloaded '*' operator for the multiplication of a matrix or
    scalar with an ellipsoid

    Syntax:
        E = factor1 * factor2
        E = mtimes(factor1,factor2)

    Inputs:
        factor1 - ellipsoid object, numeric matrix/scalar
        factor2 - ellipsoid object, numeric scalar

    Outputs:
        E - ellipsoid object

    Example: 
        E = ellipsoid([2.7 -0.2;-0.2 2.4])
        M = [1 0.5; 0.5 1]
 
        figure hold on
        plot(E,[1,2],'b')
        plot(M*E,[1,2],'r')

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: plus

    Authors:       Victor Gassmann (MATLAB)
                   Python translation by AI Assistant
    Written:       13-March-2019 (MATLAB)
    Last update:   15-October-2019 (MATLAB)
                   07-June-2022 (avoid construction of new ellipsoid object, MATLAB)
                   04-July-2022 (VG, input checks, allow E to be class array, MATLAB)
                   05-October-2024 (MW, remove class array, MATLAB)
    Python translation: 2025
    """
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
    from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck

    try:
        # matrix/scalar * ellipsoid
        if isinstance(factor1, np.ndarray) or np.isscalar(factor1):
            # -> factor2 must be an ellipsoid object
            E = factor2.copy()
            # empty set: result is empty set
            if E.representsa_('emptySet', E.TOL):
                return E
            # compute auxiliary value for new shape matrix
            M = factor1 @ E.Q @ factor1.T
            # make sure it is symmetric
            M = 0.5 * (M + M.T)
            
            E.Q = M
            E.q = factor1 @ E.q
            return E
        
        # ellipsoid * scalar
        if isinstance(factor2, np.ndarray) or np.isscalar(factor2):
            # -> factor1 must be an ellipsoid object
            E = factor1.copy()
            # empty set: result is empty set
            if E.representsa_('emptySet', E.TOL):
                return E
            # compute auxiliary value for new shape matrix
            M = factor2**2 * E.Q
            # make sure it is symmetric
            M = 0.5 * (M + M.T)
            
            E.Q = M
            E.q = factor2 * E.q
            return E

    except Exception as ME:
        # check whether different dimension of ambient space
        try:
            equalDimCheck(factor1, factor2)
        except:
            pass
        raise ME

    raise TypeError(f"Operation '*' not supported between instances of '{type(factor1).__name__}' and '{type(factor2).__name__}'") 