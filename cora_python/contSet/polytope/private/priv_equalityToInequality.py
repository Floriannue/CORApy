import numpy as np
from typing import Tuple

def priv_equalityToInequality(A: np.ndarray, b: np.ndarray, Ae: np.ndarray, be: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rewrites all equality constraints as inequality constraints.
    Ae*x = be  =>  Ae*x <= be and -Ae*x <= -be
    """
    
    A_new = A
    b_new = b
    
    if Ae is not None and Ae.size > 0 and be is not None and be.size > 0:
        if A_new is None or A_new.size == 0:
             A_new = np.vstack([Ae, -Ae])
             b_new = np.vstack([be, -be])
        else:
             A_new = np.vstack([A, Ae, -Ae])
             b_new = np.vstack([b, be, -be])

    return A_new, b_new 