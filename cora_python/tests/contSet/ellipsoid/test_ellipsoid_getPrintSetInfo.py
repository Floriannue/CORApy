import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.getPrintSetInfo import getPrintSetInfo


def test_ellipsoid_getPrintSetInfo_basic():
    """Test basic getPrintSetInfo functionality for Ellipsoid."""
    Q = np.eye(2)
    E = Ellipsoid(Q)
    abbrev, propertyOrder = getPrintSetInfo(E)
    assert abbrev == 'E', "Abbreviation should be 'E'"
    assert propertyOrder == ['Q', 'q', 'TOL'], "Property order is incorrect"

def test_ellipsoid_getPrintSetInfo_empty():
    """Test getPrintSetInfo for an empty Ellipsoid."""
    Q_empty = np.array([[]]).reshape(0,0)
    E_empty = Ellipsoid(Q_empty, np.array([]).reshape(0,1))
    abbrev, propertyOrder = getPrintSetInfo(E_empty)
    assert abbrev == 'E', "Abbreviation should be 'E' for empty ellipsoid"
    assert propertyOrder == ['Q', 'q', 'TOL'], "Property order is incorrect for empty ellipsoid" 