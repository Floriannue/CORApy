import numpy as np
import scipy.sparse

def hessianTensor_jetEngine(x, u):

    Hf = [None] * 2


    # Use lil_matrix for efficient element assignment
    Hf[0] = scipy.sparse.lil_matrix((3, 3))
    Hf[0][0, 0] = -3.0*x[0, 0] - 3.0

    Hf[1] = scipy.sparse.lil_matrix((3, 3))


    Hg = []


    return Hf
