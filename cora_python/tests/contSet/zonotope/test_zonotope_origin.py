"""
test_zonotope_origin - unit test function of origin

Syntax:
    python -m pytest test_zonotope_origin.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 21-September-2024 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_zonotope_origin():
    """Unit test function of origin - mirrors MATLAB test_zonotope_origin.m"""
    
    # 1D
    Z = Zonotope.origin(1)
    Z_true = Zonotope(np.array([[0]]))
    assert Z.isequal(Z_true)
    res, cert, scaling = Z.contains_(np.array([[0]]))
    assert res
    
    # 2D
    Z = Zonotope.origin(2)
    Z_true = Zonotope(np.array([[0], [0]]))
    assert Z.isequal(Z_true)
    res, cert, scaling = Z.contains_(np.array([[0], [0]]))
    assert res
    
    # wrong calls
    with pytest.raises(CORAerror):
        Zonotope.origin(0)
    with pytest.raises(CORAerror):
        Zonotope.origin(-1)
    with pytest.raises(CORAerror):
        Zonotope.origin(0.5)
    with pytest.raises(CORAerror):
        Zonotope.origin([1, 2])
    with pytest.raises(CORAerror):
        Zonotope.origin("text") 