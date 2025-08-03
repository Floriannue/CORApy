"""
Test file for 2D zonotope vertices method - translated from MATLAB

This was formerly known as zonotope/polygon

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 28-April-2023 (MATLAB)
Last update: 11-October-2024 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.zonotope.vertices_ import vertices_
from cora_python.g.functions.matlab.validate.check import compareMatrices


def test_zonotope_vertices_2D():
    """Test 2D zonotope vertices method - translated from MATLAB"""
    
    # Empty zonotope
    Z = Zonotope.empty(2)
    p = vertices_(Z)
    assert p.size == 0 and p.shape == (2, 0)
    
    for i in range(1, 3):
        if i == 1:
            # Non-degenerate zonotope
            c = np.array([1, -1])
            G = np.array([[2, -3, 1, 0], [-1, 1, 0, 2]])
            p_true = np.array([[-5, 3, 3, 1, 7, -1, 7, -5, 5, -5, -1, -3, -5, -1], 
                              [-3, 3, 1, 1, -1, -1, -5, -5, -5, -5, -3, -3, -1, -1]])
        elif i == 2:
            # Degenerate zonotope
            c = np.array([2, -1])
            G = np.array([[1, 0], [0, 0]])
            p_true = np.array([[1, 3], [-1, -1]])
        
        Z = Zonotope(c, G)
        
        # Compute polygon points
        p = vertices_(Z)
        
        # Compare to true points
        
        # True points are a subset because polygon is closed
        assert compareMatrices(p_true, p, 0, 'subset')
        # No differing points
        p_unique = np.unique(p.T, axis=0).T
        assert compareMatrices(p_true, p_unique)
        
        # Check whether ordering is correct
        angles = np.degrees(np.arctan2(p[1, :], p[0, :]))
        idx_max = np.argmax(angles)
        angles_ = np.concatenate([angles[idx_max+1:], angles[:idx_max+1]])
        assert np.all(np.diff(angles_) >= 0)


if __name__ == '__main__':
    test_zonotope_vertices_2D()
    print("All tests passed!") 