"""
minkDiff - compute the Minkowski difference of two polytopes:
        P - S = P_out <-> P_out + S \subseteq P

Syntax:
    P_out = minkDiff(P,S)
    P_out = minkDiff(P,S,type)

Inputs:
    P - polytope object
    S - ContSet object, or numerical vector
    type - type of computation ('exact': default, 'exact:vertices')

Outputs:
    P_out - polytope object after Minkowski difference

Authors:       Niklas Kochdiker (MATLAB)
               Python translation by AI Assistant
Written:       04-February-2021 (MATLAB)
Last update:   14-July-2024 (MW, bug fix for equality constraints) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, List, Tuple, Any, TYPE_CHECKING

from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck

from cora_python.contSet.polytope.private.priv_minkdiff import priv_minkdiff # Our new helper

if TYPE_CHECKING:
    from .polytope import Polytope
    from cora_python.contSet.contSet.contSet import ContSet # For type hints when S is ContSet
    # Dynamic imports for other set types (Zonotope, Interval etc.) will be handled in logic

def minkDiff(P: 'Polytope', S: Union[Any, np.ndarray], type: str = 'exact') -> 'Polytope':
    """
    Computes the Minkowski difference of two polytopes or a polytope and a numerical vector.
    """
    # 0. set default values
    # Handled by default argument in Python function signature for 'type'

    # 1. check input arguments
    inputArgsCheck([
        [P, 'att', 'polytope'],
        [S, 'att', {'contSet', 'numeric'}],
        [type, 'str', {'exact', 'exact:vertices'}]
    ])

    # 2. different algorithms for different set representations
    if isinstance(S, (int, float, np.ndarray)):
        # MATLAB: P_out = P + (-S);
        # Need to ensure -S is handled correctly for numeric types.
        # Polytope.__add__ (plus) should handle P + numeric.
        # This would implicitly call P.plus(P, -S)
        # It's better to ensure P is on left side as it has type information
        from cora_python.contSet.polytope.plus import plus # Direct import
        return plus(P, -S)
    
    # 3. check dimensions
    equalDimCheck(P, S) # Throws error if dimensions are not compatible

    # 4. read out dimension
    n = P.dim()

    P_ = P.compact_(method='all', tol=1e-9) # Explicitly pass method and tol
    print(f"DEBUG (_aux_minkDiff_1D): P_.dim() after compact_: {P_.dim()}")

    # 5. fullspace minuend
    # MATLAB: if representsa_(P,'fullspace',0)
    if P_.representsa_('fullspace', 0):
        from cora_python.contSet.polytope.Inf import Inf # Static method
        return Inf(n)

    # 6. 1D case for polytopes
    # MATLAB: if n == 1 && isa(S,'polytope')
    if n == 1 and isinstance(S, P.__class__): # Check if S is also a Polytope
        # Need to implement aux_minkDiff_1D
        return _aux_minkDiff_1D(P, S)

    # 7. exact computation (default)
    if type == 'exact':
        # only requires support function evaluation of subtrahend, see [1, (27)]

        # ensure that halfspace representation is there
        P.constraints() # Computes and sets P.A, P.b, P.Ae, P.be if not available

        # compute the Minkowski difference (including emptiness check)
        # [A,b,Ae,be,empty] = priv_minkDiff(P.A_.val,P.b_.val,P.Ae_.val,P.be_.val,S);
        # Using properties which internally call .val if necessary, and handle empty states correctly.
        A_out, b_out, Ae_out, be_out, empty = priv_minkdiff(P.A, P.b, P.Ae, P.be, S)

        # init resulting polytope
        # MATLAB: polytope.empty(n) or polytope(A,b,Ae,be)
        # Ensure we use Polytope class for construction from this module
        from .polytope import Polytope
        if empty == True: # Explicit check
            returned_polytope = Polytope.empty(n)
        else:
            print(f"DEBUG (minkDiff): A_out shape: {A_out.shape}, b_out shape: {b_out.shape}")
            print(f"DEBUG (minkDiff): Ae_out shape: {Ae_out.shape}, be_out shape: {be_out.shape}")
            print(f"DEBUG (minkDiff): A_out content:\n{A_out}")
            print(f"DEBUG (minkDiff): b_out content:\n{b_out}")
            print(f"DEBUG (minkDiff): Ae_out content:\n{Ae_out}")
            print(f"DEBUG (minkDiff): be_out content:\n{be_out}")
            # Need to handle case where Ae_out or be_out are empty but Polytope constructor expects them.
            # The constructor should handle 0xN or 0x1 matrices correctly for empty equality constraints.
            returned_polytope = Polytope(A_out, b_out, Ae=Ae_out, be=be_out)
            if Ae_out.size == 0 and be_out.size == 0:
                returned_polytope = Polytope(A_out, b_out)
            else:
                returned_polytope = Polytope(A_out, b_out, Ae=Ae_out, be=be_out)
            print(f"DEBUG (minkDiff): ID of returned Polytope: {id(returned_polytope)}")
        return returned_polytope

    elif type == 'exact:vertices':
        # This part is more complex and will be implemented later if required
        # It involves conversions to other set types, Minkowski additions/intersections
        # For now, raise not supported error as per existing pattern for unimplemented features.
        raise CORAerror('CORA:notSupported','minkDiff with type \'exact:vertices\' is not yet translated.')

    # 8. set properties (aux_setproperties)
    # This function is not called in MATLAB if type is 'exact:vertices'
    # It's called after the main computation paths.
    P_out = _aux_setproperties(P_out, P, S) # P_out should be the result from above
    return P_out


# Auxiliary functions -----------------------------------------------------

def _aux_minkDiff_1D(P: 'Polytope', S: 'Polytope') -> 'Polytope':
    """
    Auxiliary function for 1D Minkowski difference between two polytopes.
    """
    # dimension
    n = P.dim()

    from .polytope import Polytope # Import Polytope for use in this auxiliary function

    P_ = P.compact_(method='all', tol=1e-9) # Explicitly pass method and tol
    print(f"DEBUG (_aux_minkDiff_1D): P_.dim() after compact_: {P_.dim()}")

    # if the minuend is fullspace, minkDiff(P1,S) = fullspace
    # MATLAB: representsa_(P_,'fullspace',0)
    if P_.representsa_('fullspace', 0):
        from cora_python.contSet.fullspace import Fullspace
        return Fullspace(n)

    # MATLAB: S_ = compact(S);
    # This is ContSet.compact(). Check its existence or create placeholder.
    S_ = S.compact()

    # if the subtrahend is empty, minkDiff(P1,S) = R^n
    if S_.representsa_('emptySet', 1e-10):
        from cora_python.contSet.fullspace import Fullspace
        return Fullspace(n)

    # check for equality constraints (not supported in MATLAB 1D minkDiff here)
    if S.Ae.size > 0 or P.Ae.size > 0: # Using properties, not .val
        raise CORAerror('CORA:notSupported', 'minkDiff for equality constraints currently not supported in 1D.')

    # both A matrices are normalized in minHRep (only 1D)
    # MATLAB: P_out = polytope(P_.A_.val, P_.b_.val - S_.b_.val);
    # A, b properties already return values, no .val needed.
    return Polytope(P_.A, P_.b - S_.b)

def _aux_setproperties(P_out: 'Polytope', P: 'Polytope', S: Any) -> 'Polytope':
    """
    Auxiliary function to set properties of the resulting polytope `P_out`.
    Currently only supports `Polytope` `S`.
    """
    # MATLAB: if isa(S,'polytope')
    if isinstance(S, P.__class__):

        # If both polytopes are bounded, then difference is also bounded
        # MATLAB: (~isempty(P.bounded.val) && P.bounded.val)
        # Python: P.isBounded is a property/method, not .val
        if P.isBounded() and S.isBounded(): # Assuming isBounded is a method returning bool
            P_out._bounded_val = True # Set private property for direct assignment

        # If one of the polytopes is unbounded, then difference is also unbounded
        # MATLAB: (~isempty(P.bounded.val) && ~P.bounded.val)
        if not P.isBounded() or not S.isBounded(): # Assuming isBounded is a method returning bool
            P_out._bounded_val = False # Set private property

        # If one of the polytopes is fully dimensional, then difference is also fully dimensional
        # MATLAB: (~isempty(P.fullDim.val) && P.fullDim.val)
        if P.isFullDim() or S.isFullDim(): # Assuming isFullDim is a method returning bool
            P_out._fullDim_val = True # Set private property

    return P_out
