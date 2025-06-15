import numpy as np

def isemptyobject(C) -> bool:
    """
    isemptyobject - checks whether a capsule contains any information at all;
    consequently, the set is interpreted as the empty set

    Syntax:
       res = isemptyobject(C)

    Inputs:
       C - capsule object

    Outputs:
       res - true/false

    Example:
       # C = capsule([1;-1],[0;1],1);
       # isemptyobject(C); # false

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Mark Wetzlinger
    Written:       24-July-2023
    Last update:   ---
    Last revision: ---
    """

    # In MATLAB, `size(C)` returns [rows, cols] for an array of objects.
    # In Python, we typically iterate through a list of objects or handle a single object.
    # Assuming C is a single Capsule object or a list of Capsule objects.
    if isinstance(C, list) or isinstance(C, np.ndarray) and C.ndim > 0:
        res = np.full(C.shape, False, dtype=bool)
        for i in np.ndindex(C.shape):
            res[i] = _aux_checkIfEmpty(C[i])
    else:
        res = _aux_checkIfEmpty(C)
    return res

def _aux_checkIfEmpty(C_obj) -> bool:
    # aux_checkIfEmpty - checks if a single Capsule object is empty based on its properties

    # In MATLAB, isnumeric(C.c) && isempty(C.c) checks if c is a numeric type and empty.
    # In Python, we check if it's a numpy array and then if its size is 0.
    # A capsule is empty if both center and generator are empty arrays
    c_empty = isinstance(C_obj.c, np.ndarray) and C_obj.c.size == 0
    g_empty = isinstance(C_obj.g, np.ndarray) and C_obj.g.size == 0
    
    # For an empty capsule, the radius doesn't matter - if c and g are empty, it's empty
    return c_empty and g_empty 