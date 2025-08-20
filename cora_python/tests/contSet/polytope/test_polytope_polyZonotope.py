import numpy as np
import pytest

from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_polytope_polyZonotope_2d_bounded():
    """Test conversion of a 2D bounded polytope to a polynomial zonotope.
    Samples points from the pZ and checks if they are contained in the original polytope.
    """
    # 2D, bounded
    A = np.array([[2, 1], [-1, 3], [-2, -2], [1, -3]])
    b = np.ones((4, 1))
    P = Polytope(A, b)

    # convert to polyZonotope
    pZ = P.polyZonotope()

    # sample random points from converted polyZonotope and check if all are
    # contained in the polytope
    # MATLAB: p_pZ = randPoint(pZ,1000);
    # Note: randPoint is an instance method attached via __init__.py
    p_pZ = pZ.randPoint_(1000)

    # MATLAB: assert(all(contains(P,p_pZ)));
    # Note: contains is an instance method attached via __init__.py
    # contains_ returns (res, cert, scaling), we only need res here.
    res_contains, _, _ = P.contains_(p_pZ)
    assert np.all(res_contains)

def test_polytope_polyZonotope_3d_fully_empty():
    """Test conversion of a 3D fully empty polytope to a polynomial zonotope.
    Should raise CORAerror:specialError.
    """
    # 3D, fully empty
    A = np.zeros((0, 3))
    b = np.zeros((0, 0)) # Should be (0,1) for column vector, but MATLAB might handle (0,0)
    P = Polytope(A, b) # This should create an empty polytope

    # MATLAB: assertThrowsAs(@polyZonotope,'CORA:specialError',P);
    with pytest.raises(CORAerror) as excinfo:
        P.polyZonotope()
    assert 'Polytope is unbounded and can therefore not be converted into a polynomial zonotope.' in str(excinfo.value)

def test_polytope_polyZonotope_2d_fullspace():
    """Test conversion of a 2D fullspace polytope (trivially fulfilled constraints)
    to a polynomial zonotope. Should raise CORAerror:specialError.
    """
    # 2D, trivially fulfilled constraints (fullspace)
    A = np.array([[0, 0]])
    b = np.array([[1]])
    Ae = np.array([[0, 0]])
    be = np.array([[0]])
    P = Polytope(A, b, Ae=Ae, be=be)

    # MATLAB: assertThrowsAs(@polyZonotope,'CORA:specialError',P);
    with pytest.raises(CORAerror) as excinfo:
        P.polyZonotope()
    assert 'Polytope is unbounded and can therefore not be converted into a polynomial zonotope.' in str(excinfo.value)
