import pytest
import numpy as np

from cora_python.contSet.zonotope.zonotope import zonotope
from cora_python.contSet.contSet.projectHighDim import projectHighDim
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check import compareMatrices

def test_contSet_projectHighDim():
    # Test 1: Basic projection to higher dimensions
    c = np.array([1, -1])
    G = np.array([[1, 0], [0, 2]])
    Z = zonotope(c, G)

    N = 4
    proj = [1, 3]  # Project to 1st and 3rd dimensions
    Z_high = Z.projectHighDim(N, proj)

    c_true = np.array([1, 0, -1, 0])
    G_true = np.array([[1, 0, 0, 0], [0, 0, 2, 0]]).T
    
    assert isinstance(Z_high, zonotope)
    assert Z_high.dim() == N
    assert compareMatrices(Z_high.c, c_true)
    assert compareMatrices(Z_high.G, G_true)

    # Test 2: Reordering dimensions
    c = np.array([1, 2, 3])
    G = np.eye(3)
    Z = zonotope(c, G)

    N = 3
    proj = [3, 1, 2] # Reorder as (z, x, y)
    Z_reorder = Z.projectHighDim(N, proj)

    c_true = np.array([2, 3, 1])
    G_true = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    
    assert compareMatrices(Z_reorder.c, c_true)
    assert compareMatrices(Z_reorder.G, G_true)

    # Test 3: Default projection
    Z = zonotope(c, G)
    Z_default = Z.projectHighDim(3) # Should be identity
    assert compareMatrices(Z_default.c, c)
    assert compareMatrices(Z_default.G, G)

    # Test 4: Error handling
    Z = zonotope(np.zeros(3), np.eye(3))

    # Error: Target dimension N is smaller than original
    with pytest.raises(CORAerror) as e:
        Z.projectHighDim(2)
    assert "Dimension of higher-dimensional space must be larger than or equal" in str(e.value)
    
    # Error: Length of proj does not match dimension
    with pytest.raises(CORAerror) as e:
        Z.projectHighDim(4, [1, 2])
    assert "Number of dimensions in higher-dimensional space must match" in str(e.value)

    # Error: proj dimension exceeds N
    with pytest.raises(CORAerror) as e:
        Z.projectHighDim(3, [1, 2, 4])
    assert "Specified dimensions exceed dimension of high-dimensional space" in str(e.value) 