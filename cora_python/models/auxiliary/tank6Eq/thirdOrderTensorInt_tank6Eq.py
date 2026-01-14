import numpy as np
import scipy.sparse
from cora_python.contSet.interval import Interval

def thirdOrderTensorInt_tank6Eq(x, u):
    Tf = [[None for _ in range(7)] for _ in range(6)]


    Tf[0][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[0][0][0, 0] = -0.0249156389141439/x[0, 0]**(5/2)

    Tf[0][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[0][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[1][0][0, 0] = 0.0249156389141439/x[0, 0]**(5/2)

    Tf[1][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[1][1][1, 1] = -0.0249156389141439/x[1, 0]**(5/2)

    Tf[1][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[1][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[2][1][1, 1] = 0.0249156389141439/x[1, 0]**(5/2)

    Tf[2][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[2][2][2, 2] = -0.0249156389141439/x[2, 0]**(5/2)

    Tf[2][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[2][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[3][2][2, 2] = 0.0249156389141439/x[2, 0]**(5/2)

    Tf[3][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[3][3][3, 3] = -0.0249156389141439/x[3, 0]**(5/2)

    Tf[3][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[3][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[4][3][3, 3] = 0.0249156389141439/x[3, 0]**(5/2)

    Tf[4][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[4][4][4, 4] = -0.0249156389141439/x[4, 0]**(5/2)

    Tf[4][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[4][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[5][0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[5][1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[5][2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[5][3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))

    Tf[5][4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[5][4][4, 4] = 0.0249156389141439/x[4, 0]**(5/2)

    Tf[5][5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tf[5][5][5, 5] = -0.0249156389141439/x[5, 0]**(5/2)

    Tf[5][6] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Tg = [[None for _ in range(1)] for _ in range(1)]

    Tg[0][0] = Interval(scipy.sparse.csr_matrix((1, 7)), scipy.sparse.csr_matrix((1, 7)))


    ind = [[] for _ in range(6)]
    ind[0] = [0]
    ind[1] = [0, 1]
    ind[2] = [1, 2]
    ind[3] = [2, 3]
    ind[4] = [3, 4]
    ind[5] = [4, 5]

    return Tf, ind
