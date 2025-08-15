"""
setVertices - insert vertex matrix into polytope (MATLAB parity)
"""

import numpy as np


def setVertices(P, V: np.ndarray):
    # Directly set vertices; caller must ensure correctness
    P.V = V
    P.isVRep = True


