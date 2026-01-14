import numpy as np
import scipy.sparse

def hessianTensor_tank6Eq(x, u):

    Hf = [None] * 6


    Hf[0] = scipy.sparse.csr_matrix((7, 7))
    Hf[0][0, 0] = 0.0166104259427626/x[0, 0]**(3/2)

    Hf[1] = scipy.sparse.csr_matrix((7, 7))
    Hf[1][0, 0] = -0.0166104259427626/x[0, 0]**(3/2)
    Hf[1][1, 1] = 0.0166104259427626/x[1, 0]**(3/2)

    Hf[2] = scipy.sparse.csr_matrix((7, 7))
    Hf[2][1, 1] = -0.0166104259427626/x[1, 0]**(3/2)
    Hf[2][2, 2] = 0.0166104259427626/x[2, 0]**(3/2)

    Hf[3] = scipy.sparse.csr_matrix((7, 7))
    Hf[3][2, 2] = -0.0166104259427626/x[2, 0]**(3/2)
    Hf[3][3, 3] = 0.0166104259427626/x[3, 0]**(3/2)

    Hf[4] = scipy.sparse.csr_matrix((7, 7))
    Hf[4][3, 3] = -0.0166104259427626/x[3, 0]**(3/2)
    Hf[4][4, 4] = 0.0166104259427626/x[4, 0]**(3/2)

    Hf[5] = scipy.sparse.csr_matrix((7, 7))
    Hf[5][4, 4] = -0.0166104259427626/x[4, 0]**(3/2)
    Hf[5][5, 5] = 0.0166104259427626/x[5, 0]**(3/2)


    Hg = []


    return Hf
