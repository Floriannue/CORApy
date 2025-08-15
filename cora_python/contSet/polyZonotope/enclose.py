import numpy as np
from typing import Union, Tuple

from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.helper.sets.contSet.polyZonotope.mergeExpMatrix import mergeExpMatrix
from cora_python.g.functions.helper.sets.contSet.polyZonotope.removeRedundantExponents import removeRedundantExponents

# External dependencies for Zonotope operations
from cora_python.contSet.zonotope.enclose import enclose as zonotope_enclose
from cora_python.contSet.zonotope.generators import generators as zonotope_generators

def enclose(pZ: PolyZonotope, *varargin) -> PolyZonotope:
    """
    enclose - encloses a polynomial zonotope and its affine transformation

    Description:
        Computes the set
        { a x1 + (1 - a) * x2 | x1 \in pZ, x2 \in pZ2, a \in [0,1] }
        where pZ2 = M*pZ + pZplus

    Syntax:
        pZ = enclose(pZ,pZ2)
        pZ = enclose(pZ,M,pZplus)

    Inputs:
        pZ - polyZonotope object
        pZ2 - polyZonotope object
        M - matrix for the linear transformation
        pZplus - polyZonotope object added to the linear transformation

    Outputs:
        pZ - polyZonotope object
    """

    n_args = len(varargin)

    if not (1 <= n_args <= 2):
        raise CORAerror('CORA:wrongInputInFunction', 'Invalid number of input arguments.')

    pZ2: PolyZonotope
    if n_args == 1:
        pZ2 = varargin[0]
    elif n_args == 2:
        M = varargin[0]
        pZplus = varargin[1]
        # This assumes PolyZonotope objects support matrix multiplication and addition.
        pZ2 = M * pZ + pZplus

    # check if exponent matrices are identical
    # In Python, np.array_equal is more robust for comparing arrays.
    # For id, ensure they are 1D arrays for comparison if they can be column vectors.
    pZ_id_flat = pZ.id.flatten()
    pZ2_id_flat = pZ2.id.flatten()

    if (pZ_id_flat.shape == pZ2_id_flat.shape and np.array_equal(pZ_id_flat, pZ2_id_flat) and
        pZ.E.shape == pZ2.E.shape and np.array_equal(pZ.E, pZ2.E)):

        # compute convex hull of the dependent generators
        G = np.hstack((0.5 * pZ.G + 0.5 * pZ2.G,
                       0.5 * pZ.G - 0.5 * pZ2.G,
                       0.5 * pZ.c - 0.5 * pZ2.c))

        c = 0.5 * pZ.c + 0.5 * pZ2.c

        temp = np.ones((1, pZ.E.shape[1])) # Should be pZ.E.shape[1] if E is 2D
        
        # Ensure E is 2D, even if 0 columns
        E_pZ = pZ.E if pZ.E.ndim == 2 else np.array(pZ.E).reshape(-1, 0)

        E_concat = np.vstack((np.hstack((E_pZ, E_pZ)),
                              np.hstack((np.zeros((1, E_pZ.shape[1]), dtype=int), temp.astype(int)))))
        E = np.hstack((E_concat, np.vstack((np.zeros((E_concat.shape[0] - 1, 1), dtype=int), np.array([[1]])) )))

        if pZ.id.size > 0:
            id_val = np.vstack((pZ.id, np.array([[np.max(pZ.id) + 1]])))
        else:
            id_val = np.array([[1]])

        # compute convex hull of the independent generators by using the
        # enclose function for linear zonotopes
        temp_zeros = np.zeros((pZ.c.shape[0], 1))
        Z1 = Zonotope(temp_zeros, pZ.GI)
        Z2 = Zonotope(temp_zeros, pZ2.GI)

        Z_enclosed = zonotope_enclose(Z1, Z2)
        GI = zonotope_generators(Z_enclosed)

        # construct resulting polynomial zonotope object
        pZ_out = PolyZonotope(c, G, GI, E)
        pZ_out.id = id_val
        return pZ_out

    else:
        # bring the exponent matrices to a common representation
        id_merged, E1_adapted, E2_adapted = mergeExpMatrix(pZ.id, pZ2.id, pZ.E, pZ2.E)

        # extend generator and exponent matrix by center vector
        G1 = np.hstack((pZ.c, pZ.G))
        G2 = np.hstack((pZ2.c, pZ2.G))

        E1_ext = np.hstack((np.zeros((E1_adapted.shape[0], 1), dtype=int), E1_adapted))
        E2_ext = np.hstack((np.zeros((E2_adapted.shape[0], 1), dtype=int), E2_adapted))

        # compute convex hull of the dependent generators
        # MATLAB: G = 0.5 * [G1, G1, G2, -G2];
        G_new = 0.5 * np.hstack((G1, G1, G2, -G2))

        h1 = E1_ext.shape[1]
        h2 = E2_ext.shape[1]
        
        # Need to ensure the dimensions for hstack and vstack are correct.
        # The MATLAB logic implies: [E1_ext, E1_ext, E2_ext, E2_ext] horizontally
        # and then a new row [zeros(1,h1), ones(1,h1), zeros(1,h2), ones(1,h2)] vertically
        E_new_top = np.hstack((E1_ext, E1_ext, E2_ext, E2_ext))
        E_new_bottom = np.hstack((np.zeros((1, h1), dtype=int),
                                  np.ones((1, h1), dtype=int),
                                  np.zeros((1, h2), dtype=int),
                                  np.ones((1, h2), dtype=int)))
        E_new = np.vstack((E_new_top, E_new_bottom))

        # MATLAB: id = [id; max(id)+1];
        # This is for the *new* id from the merged expression matrix.
        id_final = np.vstack((id_merged, np.array([[np.max(id_merged) + 1]])))

        # compute convex hull of the independent generators by using the
        # enclose function for linear zonotopes
        temp_zeros = np.zeros((pZ.c.shape[0], 1))
        Z1 = Zonotope(temp_zeros, pZ.GI)
        Z2 = Zonotope(temp_zeros, pZ2.GI)

        Z_enclosed = zonotope_enclose(Z1, Z2)
        GI = zonotope_generators(Z_enclosed)

        # add up all generators that belong to identical exponents
        Enew_final, Gnew_final = removeRedundantExponents(E_new, G_new)

        # extract the center vector
        # Find columns where all exponents sum to zero
        sum_Enew = np.sum(Enew_final, axis=0)
        ind = np.where(sum_Enew == 0)[0]

        c = np.sum(Gnew_final[:, ind], axis=1, keepdims=True) # Sum columns and keep as column vector
        
        # Remove the columns used for center from Gnew_final and Enew_final
        Gnew_final = np.delete(Gnew_final, ind, axis=1)
        Enew_final = np.delete(Enew_final, ind, axis=1)

        # construct resulting polynomial zonotope object
        pZ_out = PolyZonotope(c, Gnew_final, GI, Enew_final)
        pZ_out.id = id_final
        return pZ_out
