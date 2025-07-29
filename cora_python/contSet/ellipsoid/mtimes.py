import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.contSet.contSet import ContSet # Import ContSet to check for its type
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check # For dimension check


def mtimes(factor1, factor2):
    """
    mtimes - Overloaded '*' operator for the multiplication of a matrix or
    scalar with an ellipsoid.

    The Python implementation uses the `@` operator for matrix multiplication.

    Syntax:
       E = factor1 @ factor2
       E = factor1 * factor2 (for scalar multiplication)

    Inputs:
       factor1 - Ellipsoid object, numeric matrix/scalar
       factor2 - Ellipsoid object, numeric scalar

    Outputs:
       E - Ellipsoid object

    Authors:       Victor Gassmann, Matthias Althoff (MATLAB)
                   Automatic python translation: Florian Nüssel BA 2025
    Written:       13-March-2019 (MATLAB)
    Last update:   15-October-2019 (MATLAB)
                   07-June-2022 (MATLAB, avoid construction of new ellipsoid object)
                   04-July-2022 (MATLAB, input checks, allow E to be class array)
                   05-October-2024 (MATLAB, remove class array)
    """

    # Check for empty set (return empty set if either factor is empty)
    # This assumes Ellipsoid has a representsa_ method
    # if factor1.representsa_('emptySet') or factor2.representsa_('emptySet'):
    #    return Ellipsoid.empty(factor1.dim())
    
    # MATLAB's mtimes handles E*scalar and Matrix*E. We will focus on Matrix*E for @ operator.
    # For scalar * E, it will be handled by __mul__ in Ellipsoid.

    if isinstance(factor1, np.ndarray) and isinstance(factor2, Ellipsoid):
        # matrix * ellipsoid
        E = factor2
        if E.representsa_('emptySet', E.TOL):
             return Ellipsoid.empty(E.dim())

        # compute auxiliary value for new shape matrix
        M = factor1 @ E.Q @ factor1.T
        # make sure it is symmetric
        M = 0.5 * (M + M.T)
        
        # Create a new ellipsoid object to avoid modifying in-place for chaining operations
        new_E = Ellipsoid(M, factor1 @ E.q, E.TOL)
        return new_E

    elif isinstance(factor1, Ellipsoid) and (isinstance(factor2, (int, float)) or (isinstance(factor2, np.ndarray) and factor2.size == 1)):
        # ellipsoid * scalar (or scalar as 1x1 array)
        E = factor1
        if E.representsa_('emptySet', E.TOL):
            return Ellipsoid.empty(E.dim())

        scalar = factor2 if isinstance(factor2, (int, float)) else factor2.item()

        M = (scalar**2) * E.Q
        M = 0.5 * (M + M.T)

        new_E = Ellipsoid(M, scalar * E.q, E.TOL)
        return new_E

    elif isinstance(factor1, Ellipsoid) and isinstance(factor2, Ellipsoid):
        # ellipsoid * ellipsoid (not supported by MATLAB mtimes for ellipsoids, usually implies inner product or Minkowski product)
        # For now, raise an error or return NotImplemented
        raise CORAerror('CORA:noops', factor1, factor2, 'Matrix multiplication between two ellipsoids is not supported by @ operator.')

    else:
        # Fallback for unsupported operations. Mimic MATLAB's error behavior if possible.
        # Try to call equalDimCheck to get the dimension mismatch error if applicable
        try:
            equal_dim_check(factor1, factor2)
        except CORAerror as e:
            raise e

        # If not a dimension mismatch, then it's a general unsupported operation
        raise CORAerror('CORA:noops', factor1, factor2) 