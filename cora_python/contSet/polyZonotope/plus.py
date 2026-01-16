import numpy as np
from typing import Union, Tuple

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck


# External dependencies: center, generators, dim, empty, polyZonotope constructor

def plus(pZ: Union[PolyZonotope, np.ndarray], S: Union[ContSet, np.ndarray]) -> PolyZonotope:
    """
    plus - Overloaded '+' operator for the Minkowski addition of a polynomial
    zonotope with another set representation or a point

    Syntax:
        pZ = pZ + S
        pZ = plus(pZ,S)

    Inputs:
        pZ - polyZonotope object, numeric
        S - contSet object, numeric

    Outputs:
        pZ - polyZonotope object after Minkowski addition
    """

    # ensure that numeric is second input argument (if applicable)
    # The reorderNumeric function expects arguments to be mutable for MATLAB-like behavior
    # In Python, we return new objects or reassign, so the function needs to be adapted
    # or we handle reordering manually here.
    # For simplicity, we'll assume the primary polyZonotope is `pZ` and the second operand is `S`
    # and reorder if pZ is numeric and S is a PolyZonotope.

    S_out: PolyZonotope # Represents the final PolyZonotope result

    # Inlined reorderNumeric logic
    if isinstance(pZ, (int, float, np.ndarray)) and isinstance(S, PolyZonotope):
        # Swap operands if pZ is numeric and S is PolyZonotope
        temp = pZ
        pZ = S
        S = temp
        # Work on a copy to avoid mutating the input object
        S_out = pZ.copy()
    elif isinstance(pZ, PolyZonotope):
        # Work on a copy to avoid mutating the input object
        S_out = pZ.copy()
    else:
        raise CORAerror('CORA:wrongInputInFunction', 'First input must be a PolyZonotope object or compatible numeric.')


    # call function with lower precedence
    # In Python, we typically don't have a direct 'precedence' property for dispatching
    # Instead, we rely on type checking and method overloading/dispatch.
    # The MATLAB code means: if S is a ContSet and its precedence is lower than pZ's,
    # then call S's plus method with pZ and S (swapped).
    # This implies that ContSet sub-classes would need to implement their own `plus` methods.
    # For now, we will assume pZ has higher precedence or will handle the common cases.

    try:
        if isinstance(S, PolyZonotope):
            # compute Minkowski sum of two PolyZonotopes
            S_out.c = S_out.c + S.c
            
            if S_out.c.size == 0:
                # If center becomes empty, the result is an empty polyZonotope
                # This case needs careful handling of dimension if c is empty but others aren't.
                # Assuming dim() correctly infers dimension even for empty sets based on initial state.
                S_out = PolyZonotope.empty(S_out.dim())
                return S_out
            
            S_out.G = np.hstack((S_out.G, S.G))
            
            # For exponent matrix, use block diagonal to combine
            # Ensure E matrices are 2D arrays, even if empty, for blkdiag
            E_out = S_out.E if S_out.E.ndim == 2 else np.array(S_out.E).reshape(-1, 0)
            E_S = S.E if S.E.ndim == 2 else np.array(S.E).reshape(-1, 0)
            
            S_out.E = np.block([ 
                [E_out, np.zeros((E_out.shape[0], E_S.shape[1]), dtype=int)],
                [np.zeros((E_S.shape[0], E_out.shape[1]), dtype=int), E_S]
            ])
            
            S_out.GI = np.hstack((S_out.GI, S.GI))
            
            max_id = 0
            if S_out.id.size > 0:
                max_id = np.max(S_out.id)
            
            # If S.id is empty, this operation max_id + S.id will fail or be meaningless.
            # We need to handle this to extend the existing IDs or generate new ones for S.
            new_ids_S = S.id
            if new_ids_S.size > 0:
                new_ids_S = new_ids_S + max_id

            S_out.id = np.vstack((S_out.id.reshape(-1, 1), new_ids_S.reshape(-1, 1)))

            return S_out

        # summand is a numeric vector (point)
        if isinstance(S, np.ndarray) and (S.ndim == 1 or (S.ndim == 2 and S.shape[1] == 1)):
            # Ensure S is a column vector for addition if it's a point
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            S_out.c = S_out.c + S
            return S_out

        # different cases for the different set representations
        # These conversions assume the respective constructors (e.g., zonotope(S)) exist
        # and that the converted objects have 'center' and 'generators' methods.
        if isinstance(S, Zonotope):
            # Assuming Zonotope.center and Zonotope.generators are methods or accessible functions
            from cora_python.contSet.zonotope.center import center as zonotope_center
            from cora_python.contSet.zonotope.generators import generators as zonotope_generators
            S_out.c = S_out.c + zonotope_center(S)
            S_out.GI = np.hstack((S_out.GI, zonotope_generators(S)))
            return S_out

        if isinstance(S, Interval):
            # Assuming interval has a zonotope conversion method
            from cora_python.contSet.interval.zonotope import zonotope as interval_to_zonotope
            zono_S = interval_to_zonotope(S)
            from cora_python.contSet.zonotope.center import center as zonotope_center
            from cora_python.contSet.zonotope.generators import generators as zonotope_generators
            S_out.c = S_out.c + zonotope_center(zono_S)
            S_out.GI = np.hstack((S_out.GI, zonotope_generators(zono_S)))
            return S_out

        # convert other set representations to polynomial zonotopes
        # This requires the polyZonotope(S) constructor to convert other types to PolyZonotope
        # The MATLAB code explicitly lists polytope, zonoBundle, conZonotope
        # This would lead to a recursive call to plus (S_out = S_out + S_converted)
        # which can be simplified if S_converted is a PolyZonotope.
        if isinstance(S, (Polytope, ZonoBundle, ConZonotope)):
            # Import these classes for type checking here, or rely on a generic ContSet check
            # For now, let's assume these imports will be handled where these classes are defined.
            # Need to ensure Polytope, ZonoBundle, ConZonotope are imported if used directly
            # Or dynamically import if type checking is complex.
            from cora_python.contSet.polytope.polytope import Polytope
            from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
            from cora_python.contSet.conZonotope.conZonotope import ConZonotope

            converted_S = PolyZonotope(S) # This is the crucial conversion call
            # Call plus again with two PolyZonotopes
            S_out = plus(S_out, converted_S)
            return S_out

    except Exception as ME:
        # already know what's going on...
        # Check if it's already a CORAerror, then re-raise.
        if isinstance(ME, CORAerror):
            raise

        # check whether different dimension of ambient space
        try:
            equalDimCheck(S_out, S)
        except CORAerror:
            # If equalDimCheck itself raised a CORAerror, re-raise it.
            raise
        
        # check for empty sets
        # Assuming representsa_ is a method of ContSet or a callable function
        # and dim is a method or property.
        from cora_python.contSet.polyZonotope.isemptyobject import isemptyobject as pz_isemptyobject
        from cora_python.contSet.polyZonotope.dim import dim as pz_dim
        
        if pz_isemptyobject(S_out) or pz_isemptyobject(S):
            # Ensure PolyZonotope.empty exists
            S_out = PolyZonotope.empty(pz_dim(S_out))
            return S_out

        # other error...
        raise ME

    raise CORAerror('CORA:noops', pZ, S)
