import numpy as np
import scipy.sparse
from cora_python.contSet.interval import Interval

def thirdOrderTensorInt_vanderPolEq(x, u):
    Tf = [[None for _ in range(3)] for _ in range(2)]


    Tf[0][0] = Interval(scipy.sparse.csr_matrix((3, 3)), scipy.sparse.csr_matrix((3, 3)))

    Tf[0][1] = Interval(scipy.sparse.csr_matrix((3, 3)), scipy.sparse.csr_matrix((3, 3)))

    Tf[0][2] = Interval(scipy.sparse.csr_matrix((3, 3)), scipy.sparse.csr_matrix((3, 3)))

    Tf[1][0] = Interval(scipy.sparse.csr_matrix((3, 3)), scipy.sparse.csr_matrix((3, 3)))
    Tf[1][0][0, 1] = -2
    Tf[1][0][1, 0] = -2

    Tf[1][1] = Interval(scipy.sparse.csr_matrix((3, 3)), scipy.sparse.csr_matrix((3, 3)))
    Tf[1][1][0, 0] = -2

    Tf[1][2] = Interval(scipy.sparse.csr_matrix((3, 3)), scipy.sparse.csr_matrix((3, 3)))
    Tg = [[None for _ in range(1)] for _ in range(1)]

    Tg[0][0] = Interval(scipy.sparse.csr_matrix((1, 3)), scipy.sparse.csr_matrix((1, 3)))


    ind = [[] for _ in range(2)]
    ind[0] = []
    ind[1] = [0, 1]

    return Tf, ind
