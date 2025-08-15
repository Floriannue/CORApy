import numpy as np
from typing import Tuple
from .removeRedundantIds import removeRedundantIds

def mergeExpMatrix(id1: np.ndarray, id2: np.ndarray, E1: np.ndarray, E2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mergeExpMatrix - Merge the ID-vectors of two polyZonotope objects
                     and adapte the exponent matrices accordingly

    Syntax:
        [id,E1,E2] = mergeExpMatrix(id1,id2,E1,E2)

    Inputs:
        id1 - ID-vector of the first polynomial zonotope
        id2 - ID-vector of the second polynomial zonotope
        E1 - exponent matrix of the first polynomial zonotope
        E2 - exponent matrix of the second polynomial zonotope

    Outputs:
        id_merged - merged ID-vector
        E1_adapted - adapted exponent matrix of the first polynomial zonotope
        E2_adapted - adapted exponent matrix of the second polynomial zonotope
    """

    # ensure uniqueness for inputs
    E1, id1 = removeRedundantIds(E1, id1)
    E2, id2 = removeRedundantIds(E2, id2)
    
    # Ensure id vectors are column vectors
    if id1.ndim == 1:
        id1 = id1.reshape(-1, 1)
    if id2.ndim == 1:
        id2 = id2.reshape(-1, 1)

    L1 = id1.shape[0]
    L2 = id2.shape[0]

    # ID vectors are identical
    # Use np.array_equal for robust comparison of arrays
    if L1 == L2 and np.array_equal(id1, id2):
        id_merged = id1
        E1_adapted = E1
        E2_adapted = E2

    # ID vectors not identical -> MERGE
    else:
        # Start with id1 as the base for merging
        id_merged = id1.copy() # Make a copy to avoid modifying original id1
        ind2 = np.zeros(id2.shape, dtype=int) # To store indices of id2 in id_merged

        for i in range(L2):
            current_id2 = id2[i, 0] # Get scalar value of current id from id2
            # Find if current_id2 exists in id_merged
            # np.where returns a tuple of arrays, check if any match
            ind_in_merged = np.where(id_merged.flatten() == current_id2)[0]

            if ind_in_merged.size == 0:
                # If not found, append to id_merged and record new index
                id_merged = np.vstack((id_merged, [[current_id2]]))
                ind2[i] = id_merged.shape[0] - 1 # 0-based index
            else:
                # If found, record existing index
                ind2[i] = ind_in_merged[0] # Take the first index if duplicates exist (shouldn't after removeRedundantIds)
        
        # MATLAB uses 1-based indexing for ind2, so we need to adjust for Python's 0-based indexing.
        # ind2 currently stores the 0-based indices of where elements of id2 are in id_merged.

        # construct the new exponent matrices
        L = id_merged.shape[0]

        # Extend E1 with zeros for new IDs from id2
        # If E1 is empty, need to handle its column dimension correctly.
        if E1.size == 0:
            E1_adapted = np.zeros((L, E1.shape[1] if E1.ndim > 1 else 0), dtype=E1.dtype)
        else:
            E1_adapted = np.vstack((E1, np.zeros((L - L1, E1.shape[1]), dtype=E1.dtype)))

        # Create a temporary matrix for E2, then fill based on ind2
        E2_temp = np.zeros((L, E2.shape[1]), dtype=E2.dtype) if E2.size > 0 else np.zeros((L, 0), dtype=int)
        
        # Map rows of E2 to their new positions in E2_temp using ind2
        if E2.size > 0:
            for i in range(L2):
                E2_temp[ind2[i,0], :] = E2[i, :]
        E2_adapted = E2_temp

    return id_merged, E1_adapted, E2_adapted
