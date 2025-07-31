"""
testLong_zonotope_convHull - unit test function of convHull (严格对齐MATLAB)

Syntax:
    python -m pytest testLong_zonotope_convHull.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 23-April-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope

class TestZonotopeConvHull:
    """Test class for zonotope convHull method (严格对齐MATLAB)"""

    def test_long_convex_hull_randomized(self):
        """严格复现MATLAB testLong_zonotope_convHull逻辑"""
        nrTests = 25
        nrRandPoints = 100
        for i in range(nrTests):
            # 随机维度和生成元数量
            n = np.random.randint(1, 9)  # 1~8
            nrGens = np.random.randint(n, 2*n+1)  # n~2n

            # 随机生成两个 zonotope
            Z1 = Zonotope.generateRandom('Dimension', n, 'NrGenerators', nrGens)
            Z2 = Zonotope.generateRandom('Dimension', n, 'NrGenerators', nrGens)

            # 计算 convex hull
            Z = Z1.convHull_(Z2)

            # 分别采样点
            p1 = Z1.randPoint_(nrRandPoints)
            p2 = Z2.randPoint_(nrRandPoints)

            # 线性组合
            lambdas = np.random.rand(1, nrRandPoints)
            p = lambdas * p1 + (1 - lambdas) * p2

            # 检查所有点都在 convex hull 内
            all_points = np.hstack([p1, p2, p])
            for j in range(all_points.shape[1]):
                assert Z.contains_(all_points[:, j]), f"Test {i+1}, point {j+1} not contained"