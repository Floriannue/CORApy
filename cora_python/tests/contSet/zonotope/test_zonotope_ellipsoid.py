"""
test_zonotope_ellipsoid - unit test function of ellipsoid

Syntax:
    res = test_zonotope_ellipsoid

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Victor Gassmann
Written:       27-July-2021
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_zonotope_ellipsoid():
    """Test zonotope ellipsoid conversion following MATLAB test structure."""
    
    # inner:norm
    Z1 = Zonotope(np.array([[-4, -3, -2], [1, 2, 3]]))
    E1o = Z1.ellipsoid()
    E1i = Z1.ellipsoid('inner:norm')
    
    # Note: randPoint functionality would need to be implemented
    # Y1i = randPoint(E1i, 2*dim(Z1), 'extreme')
    
    # For now, test basic ellipsoid creation
    assert isinstance(E1o, Ellipsoid)
    assert isinstance(E1i, Ellipsoid)
    
    # outer:exact / inner:exact
    Z2 = Zonotope(np.array([[1, 4, 2, 1], [-3, 2, -1, 1]]))
    E2o = Z2.ellipsoid('outer:exact')
    E2i = Z2.ellipsoid('inner:exact')
    
    # Note: randPoint functionality would need to be implemented
    # Y2i = randPoint(E2i, 2*dim(Z2), 'extreme')
    
    # For now, test basic ellipsoid creation
    assert isinstance(E2o, Ellipsoid)
    assert isinstance(E2i, Ellipsoid)
    
    # check point
    c = np.array([[3], [4], [2]])
    Z = Zonotope(c)
    E = Z.ellipsoid()
    
    # Test that it's a point ellipsoid (zero shape matrix)
    assert isinstance(E, Ellipsoid)
    np.testing.assert_array_almost_equal(E.q, c)
    np.testing.assert_array_almost_equal(E.Q, np.zeros((3, 3)))
    
    # gather results
    res = True
    assert res 