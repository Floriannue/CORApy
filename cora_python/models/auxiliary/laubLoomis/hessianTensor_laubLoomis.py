import numpy as np
import scipy.sparse

def hessianTensor_laubLoomis(x, u):

    Hf = [None] * 7


    # Use lil_matrix for efficient element assignment
    Hf[0] = scipy.sparse.lil_matrix((8, 8))

    Hf[1] = scipy.sparse.lil_matrix((8, 8))

    Hf[2] = scipy.sparse.lil_matrix((8, 8))
    Hf[2][1, 2] = -0.800000000000000
    Hf[2][2, 1] = -0.800000000000000

    Hf[3] = scipy.sparse.lil_matrix((8, 8))
    Hf[3][2, 3] = -1.30000000000000
    Hf[3][3, 2] = -1.30000000000000

    Hf[4] = scipy.sparse.lil_matrix((8, 8))
    Hf[4][3, 4] = -1.00000000000000
    Hf[4][4, 3] = -1.00000000000000

    Hf[5] = scipy.sparse.lil_matrix((8, 8))

    Hf[6] = scipy.sparse.lil_matrix((8, 8))
    Hf[6][1, 6] = -1.50000000000000
    Hf[6][6, 1] = -1.50000000000000


    Hg = []


    return Hf
