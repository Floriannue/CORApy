import numpy as np

def removeRedundantIds(E: np.ndarray, id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    removeRedundantIds - finds and removes redundant elements in the
    ID-vectors and sums the corresponding rows of the exponent matrix.

    Syntax:
        [id,E1,E2] = removeRedundantIds(E,id)

    Inputs:
        id - ID-vector of the polynomial zonotope
        E  - exponent matrix of the polynomial zonotope

    Outputs:
        id_unique - merged ID-vector
        E_unique - adapted exponent matrix
    """

    # ensure id is a 1D array for unique operation, then reshape to column vector
    id_flat = id.flatten()
    
    # find unique ids (+ indices in ID-vector)
    # np.unique with return_index=True and return_inverse=True can give us similar info to MATLAB's unique(..., 'stable')
    # However, to sum corresponding rows, a direct approach with `unique` on `id` and then iterating or using `groupby` is better.
    
    # Get unique IDs in stable order
    id_unique, unique_indices = np.unique(id_flat, return_index=True)
    id_unique = id_unique[np.argsort(unique_indices)]
    
    # If id is a column vector, make id_unique a column vector as well
    if id.ndim == 2 and id.shape[1] == 1:
        id_unique = id_unique.reshape(-1, 1)

    # check if exponent matrix is empty
    if E.size == 0:
        # Ensure E_unique has correct shape if E was an empty (0,0) or (n,0) array
        if E.ndim == 2:
            E_unique = np.zeros((len(id_unique), E.shape[1]), dtype=E.dtype)
        else:
            E_unique = np.array([]) # Default to 1D empty array
        return E_unique, id_unique

    # initialize unique exponent matrix
    E_unique = np.zeros((len(id_unique), E.shape[1]), dtype=E.dtype)

    # iterate over unique ids to sum corresponding exponents
    # This is effectively a group-by-sum operation.
    for ii, current_id in enumerate(id_unique.flatten()):
        # Find all rows in E where the corresponding id matches current_id
        matching_rows = E[id_flat == current_id, :]
        E_unique[ii, :] = np.sum(matching_rows, axis=0)

    return E_unique, id_unique
