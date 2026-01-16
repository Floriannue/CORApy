import numpy as np
import scipy.sparse

def hessianTensor_fiveDimSysEq(x, u):

    Hf = [None] * 5


    Hf[0] = scipy.sparse.csr_matrix((10, 10))

    Hf[1] = scipy.sparse.csr_matrix((10, 10))

    Hf[2] = scipy.sparse.csr_matrix((10, 10))

    Hf[3] = scipy.sparse.csr_matrix((10, 10))

    Hf[4] = scipy.sparse.csr_matrix((10, 10))


    Hg = []


    return Hf
