import numpy as np
import scipy.sparse
from cora_python.contSet.interval import Interval

def thirdOrderTensorInt_laubLoomis(x, u):
    Tf = [[None for _ in range(8)] for _ in range(7)]


    Tf[0][0] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[0][1] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[0][2] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[0][3] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[0][4] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[0][5] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[0][6] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[0][7] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[1][0] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[1][1] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[1][2] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[1][3] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[1][4] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[1][5] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[1][6] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[1][7] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[2][0] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[2][1] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[2][2] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[2][3] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[2][4] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[2][5] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[2][6] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[2][7] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[3][0] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[3][1] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[3][2] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[3][3] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[3][4] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[3][5] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[3][6] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[3][7] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[4][0] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[4][1] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[4][2] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[4][3] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[4][4] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[4][5] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[4][6] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[4][7] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[5][0] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[5][1] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[5][2] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[5][3] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[5][4] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[5][5] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[5][6] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[5][7] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[6][0] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[6][1] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[6][2] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[6][3] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[6][4] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[6][5] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[6][6] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))

    Tf[6][7] = Interval(scipy.sparse.csr_matrix((8, 8)), scipy.sparse.csr_matrix((8, 8)))
    Tg = [[None for _ in range(1)] for _ in range(1)]

    Tg[0][0] = Interval(scipy.sparse.csr_matrix((1, 8)), scipy.sparse.csr_matrix((1, 8)))


    ind = [[] for _ in range(7)]
    ind[0] = []
    ind[1] = []
    ind[2] = []
    ind[3] = []
    ind[4] = []
    ind[5] = []
    ind[6] = []

    return Tf, ind
