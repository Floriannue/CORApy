import numpy as np
import scipy.sparse
from cora_python.contSet.interval import Interval

def hessianTensorInt_fiveDimSysEq(x, u):

    Hf = [None] * 5


    Hf[0] = Interval(scipy.sparse.csr_matrix((10, 10)), scipy.sparse.csr_matrix((10, 10)))

    Hf[1] = Interval(scipy.sparse.csr_matrix((10, 10)), scipy.sparse.csr_matrix((10, 10)))

    Hf[2] = Interval(scipy.sparse.csr_matrix((10, 10)), scipy.sparse.csr_matrix((10, 10)))

    Hf[3] = Interval(scipy.sparse.csr_matrix((10, 10)), scipy.sparse.csr_matrix((10, 10)))

    Hf[4] = Interval(scipy.sparse.csr_matrix((10, 10)), scipy.sparse.csr_matrix((10, 10)))


    Hg = []


    return Hf
