import numpy as np
import scipy.sparse
from cora_python.contSet.interval import Interval

def hessianTensorInt_jetEngine(x, u):

    Hf = [None] * 2


    Hf[0] = Interval(scipy.sparse.csr_matrix((3, 3)), scipy.sparse.csr_matrix((3, 3)))
    Hf[0][0, 0] = -3.0*x[0, 0] - 3.0

    Hf[1] = Interval(scipy.sparse.csr_matrix((3, 3)), scipy.sparse.csr_matrix((3, 3)))


    Hg = []


    return Hf
