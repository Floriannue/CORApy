"""
test_nn_nnIdentityLayer - tests constructor of nnIdentityLayer

Syntax:
    res = test_nn_nnIdentityLayer()

Inputs:
    -

Outputs:
    res - boolean 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       02-October-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'cora_python'))

from cora_python.nn.layers.linear.nnIdentityLayer import nnIdentityLayer
from cora_python.contSet.zonotope import Zonotope


def test_nn_nnIdentityLayer():
    """Test constructor of nnIdentityLayer"""
    
    # simple example
    layer = nnIdentityLayer()
    
    # check evaluate
    
    # check point
    x = np.array([[1], [2], [3], [4]])
    y = layer.evaluate(x)
    
    assert np.allclose(x, y)
    
    # check zonotope
    X = Zonotope(x, 0.01 * np.eye(4))
    Y = layer.evaluate(X)
    
    assert np.allclose(X.c, Y.c) and np.allclose(X.G, Y.G)
    
    # gather results
    res = True
    return res


if __name__ == "__main__":
    test_nn_nnIdentityLayer()
    print("test_nn_nnIdentityLayer successful")
