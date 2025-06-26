import cora_python.contSet as contSet

def recompose(S):
    """
    recompose - Overload for the global recompose function.

    For the base contSet class, this method does nothing but return a
    copy of the set, as there is nothing to recompose. Subclasses
    are expected to override this method if they support decomposition
    and recomposition.

    Syntax:
        S_out = recompose(S)

    Inputs:
        S (contSet): A contSet object.

    Outputs:
        contSet: A copy of the input contSet object.
    """
    # Polymorphic dispatch
    if type(S).recompose is not contSet.contSet.ContSet.recompose:
        return type(S).recompose(S)
    
    # --- Primary Method Body ---
    
    # For the base class, nothing to recompose
    return S.copy() 