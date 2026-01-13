import numpy as np
import scipy.sparse
from cora_python.contSet.interval import Interval

def hessianTensorInt_tank(x, u):

    Hf[0] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Hf[0][0, 0] = 0.0166104259427626/x(1)**(3/2)

    Hf[1] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Hf[1][0, 0] = -0.0166104259427626/x(1)**(3/2)
    Hf[1][1, 1] = 0.0166104259427626/x(2)**(3/2)

    Hf[2] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Hf[2][1, 1] = -0.0166104259427626/x(2)**(3/2)
    Hf[2][2, 2] = 0.0166104259427626/x(3)**(3/2)

    Hf[3] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Hf[3][2, 2] = -0.0166104259427626/x(3)**(3/2)
    Hf[3][3, 3] = 0.0166104259427626/x(4)**(3/2)

    Hf[4] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Hf[4][3, 3] = -0.0166104259427626/x(4)**(3/2)
    Hf[4][4, 4] = 0.0166104259427626/x(5)**(3/2)

    Hf[5] = Interval(scipy.sparse.csr_matrix((7, 7)), scipy.sparse.csr_matrix((7, 7)))
    Hf[5][4, 4] = -0.0166104259427626/x(5)**(3/2)
    Hf[5][5, 5] = 0.0166104259427626/x(6)**(3/2)


    Hg = []


    return Hf
