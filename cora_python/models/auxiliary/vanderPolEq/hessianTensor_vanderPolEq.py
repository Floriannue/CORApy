import numpy as np
import scipy.sparse

def hessianTensor_vanderPolEq(x, u):

    Hf = [None] * 2


    Hf[0] = scipy.sparse.csr_matrix((3, 3))

    Hf[1] = scipy.sparse.csr_matrix((3, 3))
    Hf[1][0, 0] = -2*x[1, 0]
    Hf[1][0, 1] = -2*x[0, 0]
    Hf[1][1, 0] = -2*x[0, 0]


    Hg = []


    return Hf
