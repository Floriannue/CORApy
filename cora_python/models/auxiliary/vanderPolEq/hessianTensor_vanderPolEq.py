import numpy as np
import scipy.sparse

def hessianTensor_vanderPolEq(x, u):

    Hf = [None] * 2


    Hf[0] = scipy.sparse.csr_matrix((3, 3))

    # Build with LIL for efficient assignment, then convert to CSR
    Hf1 = scipy.sparse.lil_matrix((3, 3))
    Hf1[0, 0] = -2 * x[1, 0]
    Hf1[0, 1] = -2 * x[0, 0]
    Hf1[1, 0] = -2 * x[0, 0]
    Hf[1] = Hf1.tocsr()


    Hg = []


    return Hf
