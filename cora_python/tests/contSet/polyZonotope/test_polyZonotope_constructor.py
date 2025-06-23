"""
test_polyZonotope_constructor - unit test function for constructor

Syntax:
    res = test_polyZonotope_constructor()

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       28-April-2023
Last update:   04-October-2024 (MW, check default properties)
Last revision: ---
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope import PolyZonotope


def test_polyZonotope_constructor():
    """Test polyZonotope constructor with various input combinations"""
    
    # empty polyZonotope
    n = 2
    pZ = PolyZonotope.empty(n)
    assert pZ.c.shape == (n, 0)
    assert pZ.G.shape == (n, 0)
    assert pZ.GI.shape == (n, 0)
    assert pZ.E.shape == (0, 0)
    assert pZ.id.shape == (0, 1)
    assert pZ.representsa('emptySet')

    # create polynomial zonotope test data
    c = np.array([[0], [0]])
    G = np.array([[2, 0, 1], [0, 2, 1]])
    GI = np.array([[0], [0.5]])
    E = np.array([[1, 0, 3], [0, 1, 1]])
    E_def = np.eye(3)
    id_arr = np.array([[5], [6]])
    id_def2 = np.array([[1], [2]])
    id_def3 = np.array([[1], [2], [3]])

    # only center
    pZ = PolyZonotope(c)
    assert pZ.G.shape == (2, 0)
    assert pZ.c.shape == (2, 1)

    # only center and dependent generator matrix
    pZ = PolyZonotope(c, G)
    assert pZ.GI.shape == (2, 0)
    assert np.array_equal(pZ.E, E_def)
    assert np.array_equal(pZ.id, id_def3)

    # center and both generator matrices
    pZ = PolyZonotope(c, G, GI)
    assert np.array_equal(pZ.E, E_def)
    assert np.array_equal(pZ.id, id_def3)

    # only independent generator matrix
    pZ = PolyZonotope(c, np.array([]).reshape(2, 0), GI)
    assert pZ.G.shape == (2, 0)
    assert pZ.E.shape == (0, 0)
    assert pZ.id.shape == (0, 1)

    # both generator matrices and exponent matrix
    pZ = PolyZonotope(c, G, GI, E)
    assert np.array_equal(pZ.id, id_def2)

    # no independent generator matrix
    pZ = PolyZonotope(c, G, np.array([]).reshape(2, 0), E)
    assert pZ.GI.shape == (2, 0)
    assert np.array_equal(pZ.id, id_def2)

    # no independent generator matrix, with identifiers
    pZ = PolyZonotope(c, G, np.array([]).reshape(2, 0), E, id_arr)
    assert pZ.GI.shape == (2, 0)

    # all input arguments
    pZ = PolyZonotope(c, G, GI, E, id_arr)
    assert np.array_equal(pZ.c, c)
    assert np.array_equal(pZ.G, G)
    assert np.array_equal(pZ.GI, GI)
    assert np.array_equal(pZ.E, E)
    assert np.array_equal(pZ.id, id_arr)

    # copy constructor
    pZ_copy = PolyZonotope(pZ)
    assert np.array_equal(pZ_copy.c, pZ.c)
    assert np.array_equal(pZ_copy.G, pZ.G)
    assert np.array_equal(pZ_copy.GI, pZ.GI)
    assert np.array_equal(pZ_copy.E, pZ.E)
    assert np.array_equal(pZ_copy.id, pZ.id)


if __name__ == "__main__":
    test_polyZonotope_constructor() 