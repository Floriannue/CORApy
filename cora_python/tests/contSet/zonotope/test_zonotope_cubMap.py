import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.cubMap import cubMap

def compare_matrices(A, B, tol=1e-12):
    return np.allclose(A, B, atol=tol)

class TestZonotopeCubMap:
    def test_mixed_multiplication(self):
        # Define zonotope
        Z = np.array([[0, 1, -1], [1, 2, 0]])
        zono = Zonotope(Z)

        # Define third-order tensor
        temp = np.array([[1, -1], [0, 2]])
        T = [[temp, temp], [temp, temp]]

        # Compute cubic map (mixed)
        Zres = cubMap(zono, zono, zono, T)

        # Define ground truth
        temp_gt = [2, 3, 1, 4, 7, 1, 0, -1, 1, 6, 9, 3, 12, 21, 3, 0, -3, 3, -2, -3, -1, -4, -7, -1, 0, 1, -1]
        Z_gt = np.vstack([temp_gt, temp_gt])

        # Compare
        assert compare_matrices(np.hstack([Zres.center, Zres.generators]), Z_gt)

    def test_cubic_multiplication(self):
        # Define zonotope
        Z = np.array([[0, 1, -1], [1, 2, 0]])
        zono = Zonotope(Z)

        # Define third-order tensor
        temp = np.array([[1, -1], [0, 2]])
        T = [[temp, temp], [temp, temp]]

        # Compute cubic map (single)
        Zres = cubMap(zono, T)

        # Define ground truth
        temp_gt = [16, 13, -1, 14, -4, 0, 21, -7, 3, -1]
        Z_gt = np.vstack([temp_gt, temp_gt])

        # Compare
        assert compare_matrices(np.hstack([Zres.center, Zres.generators]), Z_gt) 