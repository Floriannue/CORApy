import numpy as np
import scipy.sparse
from cora_python.contSet.interval import Interval

def thirdOrderTensorInt_fiveDimSysEq(x, u):
    Tf = [[None for _ in range(10)] for _ in range(5)]


    Tf[0][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][7] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][8] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][9] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][7] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][8] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][9] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][7] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][8] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][9] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][7] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][8] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][9] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][7] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][8] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][9] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tg = [[None for _ in range(1)] for _ in range(1)]

    Tg[0][0] = Interval(scipy.sparse.csr_matrix((1, 10)), scipy.sparse.csr_matrix((1, 10)))


    ind = [[] for _ in range(5)]
    ind[0] = []
    ind[1] = []
    ind[2] = []
    ind[3] = []
    ind[4] = []

    return Tf, ind
