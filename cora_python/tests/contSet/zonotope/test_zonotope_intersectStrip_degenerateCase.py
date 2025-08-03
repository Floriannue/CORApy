"""
test_zonotope_intersectStrip_degenerateCase - unit test function of 
intersectStrip to check whether a degenerate result is obtained.
In a previous version, the method 'bravo' in [1] produced a result of 
length 1e16.

References:
    [1] Ye Wang, Vicenç Puig, and Gabriela Cembrano. Set-
        membership approach and Kalman observer based on
        zonotopes for discrete-time descriptor systems. Automatica,
        93:435-443, 2018.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       23-December-2020 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope


def test_zonotope_intersectStrip_degenerateCase():
    """Test degenerate case handling in intersectStrip"""
    # Simple 2D example which can be easily visualized
    # Zonotope
    Z = Zonotope(
        np.array([[0.0070971205961507], [-0.0060883901920886]]),
        np.array([[0.0100000000000000, -0.0200000000000000, -0.0000000000000000, 0.0340000000000000, 0.0420000000000000, 0.0160000000000000, -0.1200000000000000],
                  [-0.0300000000000000, -0.0400000000000000, 0.0000000000000000, -0.1020000000000000, -0.1260000000000000, -0.0480000000000000, 0.0200000000000000]])
    )
    
    # Strip
    C = np.array([[-2, 1]])
    y = np.array([[-0.1265718260823974]])
    phi = np.array([[0.2]])
    
    # Obtain over-approximative zonotope after intersection
    Z_over = Z.intersectStrip(C, phi, y, 'bravo')
    
    # Check whether result is not too large
    box = Z_over.interval()
    
    # Set enclosing box
    box_encl = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
    
    # Check enclosure
    assert box_encl.contains_(box)
    
    res = True
    assert res


if __name__ == "__main__":
    test_zonotope_intersectStrip_degenerateCase() 