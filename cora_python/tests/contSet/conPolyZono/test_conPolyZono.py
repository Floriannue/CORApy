"""
Unit tests for the ConPolyZono class constructor.
"""

import pytest
import numpy as np
from cora_python.contSet.conPolyZono import ConPolyZono
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_constructor_example():
    """Test example in docstring"""
    c = np.array([[0], [0]])
    G = np.array([[1, 0, 1, -1], [0, 1, 1, 1]])
    E = np.array([[1, 0, 1, 2], [0, 1, 1, 0], [0, 0, 1, 1]])
    A = np.array([[1, -0.5, 0.5]])
    b = np.array([0.5])
    EC = np.array([[0, 1, 2], [1, 0, 0], [0, 1, 0]])
    
    cPZ = ConPolyZono(c, G, E, A, b, EC)
    
    assert isinstance(cPZ, ConPolyZono)

def test_constructor_variants():
    """Test variants in syntax of docstring"""
    c = np.array([[0], [0]])
    G = np.array([[1, 0, 1, -1], [0, 1, 1, 1]])
    E = np.array([[1, 0, 1, 2], [0, 1, 1, 0], [0, 0, 1, 1]])
    GI = np.array([[4, 1], [0, 2]])
    id = np.array([[1], [2], [4]])
    A = np.array([[1, -0.5, 0.5]])
    b = np.array([0.5])
    EC = np.array([[0, 1, 2], [1, 0, 0], [0, 1, 0]])
    
    cPZ = ConPolyZono(c, G, E)
    cPZ = ConPolyZono(c, G, E, GI)
    cPZ = ConPolyZono(c, G, E, GI, id)
    cPZ = ConPolyZono(c, G, E, A, b, EC)
    cPZ = ConPolyZono(c, G, E, A, b, EC, GI)
    cPZ = ConPolyZono(c, G, E, A, b, EC, GI, id)

def test_copy_constructor():
    """Test copy constructor"""
    c = np.array([[0], [0]])
    G = np.array([[1, 0, 1, -1], [0, 1, 1, 1]])
    E = np.array([[1, 0, 1, 2], [0, 1, 1, 0], [0, 0, 1, 1]])
    A = np.array([[1, -0.5, 0.5]])
    b = np.array([0.5])
    EC = np.array([[0, 1, 2], [1, 0, 0], [0, 1, 0]])
    cPZ1 = ConPolyZono(c, G, E, A, b, EC)
    
    cPZ2 = ConPolyZono(cPZ1)
    assert isinstance(cPZ2, ConPolyZono)

def test_empty_set():
    """Test empty set"""
    cPZ = ConPolyZono.empty(2)
    assert isinstance(cPZ, ConPolyZono) 