import numpy as np
import scipy.sparse
from cora_python.contSet.interval import Interval

def hessianTensorInt_system_dynamics(x, u):

    Hf = [None] * 3


    Hf[0] = Interval(scipy.sparse.csr_matrix((4, 4)), scipy.sparse.csr_matrix((4, 4)))
    Hf[0][1, 2] = -1
    Hf[0][2, 1] = -1

    Hf[1] = Interval(scipy.sparse.csr_matrix((4, 4)), scipy.sparse.csr_matrix((4, 4)))

    Hf[2] = Interval(scipy.sparse.csr_matrix((4, 4)), scipy.sparse.csr_matrix((4, 4)))
    Hf[2][0, 1] = -1
    Hf[2][1, 0] = -1


    Hg = []


    return Hf
