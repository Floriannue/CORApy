import numpy as np
from .interval import Interval
import scipy.sparse

def subsasgn(I, S, val):
    """
    subsasgn - Overloads the operator that writes elements, e.g., I(1,2)=val,
    where the element of the first row and second column is referred to.

    Syntax:
        I = subsasgn(I,S,val)

    Inputs:
        I - interval object
        S - contains information of the type and content of element selections
        val - value to be inserted

    Outputs:
        I - interval object
    """

    # check if value is an interval
    if not isinstance(val, Interval):
        val = Interval(val, val)

    # Handle sparse matrices efficiently
    # csr_matrix and csc_matrix don't support efficient element assignment
    # Convert to lil_matrix (List of Lists) for efficient assignment
    # Keep as lil_matrix - no need to convert back unless specifically required
    if scipy.sparse.issparse(I.inf) and not isinstance(I.inf, scipy.sparse.lil_matrix):
        I.inf = I.inf.tolil()
    
    if scipy.sparse.issparse(I.sup) and not isinstance(I.sup, scipy.sparse.lil_matrix):
        I.sup = I.sup.tolil()
    
    # Extract scalar values from val if needed
    # val.inf and val.sup should be scalars, but handle sparse/array case
    val_inf = val.inf
    val_sup = val.sup
    if scipy.sparse.issparse(val.inf):
        val_inf = val.inf.toarray().item() if val.inf.size == 1 else val.inf.toarray()
    elif hasattr(val_inf, "shape") and val_inf.size == 1:
        val_inf = float(np.asarray(val_inf).flat[0])
    if scipy.sparse.issparse(val.sup):
        val_sup = val.sup.toarray().item() if val.sup.size == 1 else val.sup.toarray()
    elif hasattr(val_sup, "shape") and val_sup.size == 1:
        val_sup = float(np.asarray(val_sup).flat[0])

    # check if parentheses are used to select elements
    I.inf[S] = val_inf
    I.sup[S] = val_sup
    
    return I 