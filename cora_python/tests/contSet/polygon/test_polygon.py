"""
Unit tests for the Polygon class constructor.
"""

import pytest
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from cora_python.contSet.polygon import Polygon
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_constructor_from_vertices():
    """Tests creating a polygon from a 2xV vertex matrix."""
    V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    pgon = Polygon(V)
    assert isinstance(pgon, Polygon)
    assert isinstance(pgon.poly, ShapelyPolygon)
    assert np.allclose(np.array(pgon.poly.exterior.coords).T, np.hstack((V, V[:, 0:1])))

def test_constructor_from_xy():
    """Tests creating a polygon from x and y coordinate vectors."""
    x = np.array([0, 1, 1, 0])
    y = np.array([0, 0, 1, 1])
    pgon = Polygon(x, y)
    assert isinstance(pgon, Polygon)
    assert isinstance(pgon.poly, ShapelyPolygon)
    V = np.vstack((x, y))
    assert np.allclose(np.array(pgon.poly.exterior.coords).T, np.hstack((V, V[:, 0:1])))

def test_empty_constructor():
    """Tests creating an empty polygon."""
    pgon = Polygon()
    assert isinstance(pgon, Polygon)
    assert pgon.poly.is_empty

def test_copy_constructor():
    """Tests creating a polygon as a copy of another."""
    V = np.array([[0, 0], [1, 0], [1, 1]]).T
    pgon1 = Polygon(V)
    pgon2 = Polygon(pgon1)
    assert isinstance(pgon2, Polygon)
    assert pgon1.poly.equals(pgon2.poly)
    # Check that it is a new object (even if shapely polygons are immutable)
    assert id(pgon1) != id(pgon2)

def test_error_wrong_input():
    """Tests that a CORAerror is raised for incorrect input dimensions."""
    with pytest.raises(CORAerror):
        Polygon(np.array([1, 2, 3]))  # Not a 2xV matrix

def test_error_dimension_mismatch():
    """Tests that a CORAerror is raised for mismatched x and y vectors."""
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    with pytest.raises(CORAerror):
        Polygon(x, y) 