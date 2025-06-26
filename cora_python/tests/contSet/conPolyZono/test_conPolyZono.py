"""
Unit tests for the ConPolyZono class constructor.
"""

import pytest
import numpy as np
from cora_python.contSet.conPolyZono import ConPolyZono
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_constructor_2d():
    """Tests the full constructor with all arguments."""
    c = np.array([0, 0])
    G = np.array([[1, 0, 1, -1], [0, 1, 1, 1]])
    E = np.array([[1, 0, 1, 2], [0, 1, 1, 0], [0, 0, 1, 1]])
    A = np.array([[1, -0.5, 0.5]])
    b = np.array([0.5])
    EC = np.array([[0, 1, 2], [1, 0, 0], [0, 1, 0]])
    
    cPZ = ConPolyZono(c, G, E, A, b, EC)
    
    assert isinstance(cPZ, ConPolyZono)
    assert np.all(cPZ.c == c.reshape(-1, 1))
    assert np.all(cPZ.G == G)
    assert np.all(cPZ.E == E)
    assert np.all(cPZ.A == A)
    assert np.all(cPZ.b == b.reshape(-1, 1))
    assert np.all(cPZ.EC == EC)

def test_constructor_with_independent_generators():
    """Tests the constructor syntax with independent generators (GI)."""
    c = np.array([1, 1])
    G = np.array([[1, 0], [0, 1]])
    E = np.array([[1, 0], [0, 1]])
    GI = np.array([[0.5], [0.5]])
    
    cPZ = ConPolyZono(c, G, E, GI)
    
    assert isinstance(cPZ, ConPolyZono)
    assert np.all(cPZ.GI == GI)

def test_constructor_with_id():
    """Tests the constructor syntax that includes an id."""
    c = np.array([0])
    G = np.array([[1]])
    E = np.array([[1]])
    GI = np.array([[0.5]])
    id_ = np.array([1, 2, 3])

    A = np.array([[1]])
    b = np.array([0])
    EC = np.array([[1]])
    cPZ = ConPolyZono(c, G, E, A, b, EC, GI, id_)
    
    assert isinstance(cPZ, ConPolyZono)
    assert np.all(cPZ.id == id_.reshape(-1, 1))

def test_copy_constructor():
    """Tests the copy constructor."""
    c = np.array([0, 0])
    G = np.array([[1, 0, 1, -1], [0, 1, 1, 1]])
    E = np.array([[1, 0, 1, 2], [0, 1, 1, 0], [0, 0, 1, 1]])
    cPZ1 = ConPolyZono(c, G, E)
    
    cPZ2 = ConPolyZono(cPZ1)
    
    assert isinstance(cPZ2, ConPolyZono)
    assert np.all(cPZ1.c == cPZ2.c)
    assert np.all(cPZ1.G == cPZ2.G)
    assert np.all(cPZ1.E == cPZ2.E)
    
    # Ensure it's a deep copy
    cPZ2.c[0] = 99
    assert not np.all(cPZ1.c == cPZ2.c)

def test_errors():
    """Tests for expected errors with invalid input."""
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    E = np.array([[1, 0, 1]])
    
    with pytest.raises(CORAerror):
        ConPolyZono(c, G, E) 