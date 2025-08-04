"""
test_zonotope_and - unit test function of and_

Syntax:
    pytest test_zonotope_and.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: testLong_zonotope_and (MATLAB)

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       09-September-2020 (MATLAB)
Last update:   ---
Python translation: Florian Nüssel BA 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.emptySet import EmptySet

# 1D (convertible to intervals)
def test_zonotope_and_1d_empty():
    """1D: intersection is empty set (MATLAB: Z1 & Z2; assert(representsa(Z_and,'emptySet')) )"""
    Z1 = Zonotope(np.array([0]), np.array([[3]]))
    Z2 = Zonotope(np.array([5]), np.array([[1]]))
    Z_and = Z1 & Z2
    assert Z_and.representsa_('emptySet')

def test_zonotope_and_1d_point():
    """1D: intersection is a single point (MATLAB: Z1 & Z2; assert(isequal(Z_and,zonotope(4))) )"""
    Z1 = Zonotope(np.array([0]), np.array([[4]]))
    Z2 = Zonotope(np.array([5]), np.array([[1]]))
    Z_and = Z1 & Z2
    expected = Zonotope(np.array([4]))
    assert Z_and.c is not None and expected.c is not None and np.allclose(Z_and.c, expected.c)
    assert Z_and.G is not None and expected.G is not None and np.allclose(Z_and.G, expected.G)

def test_zonotope_and_1d_full():
    """1D: intersection is full-dimensional zonotope (MATLAB: Z1 & Z2; assert(isequal(Z_and,zonotope(4.5,0.5))) )"""
    Z1 = Zonotope(np.array([0]), np.array([[5]]))
    Z2 = Zonotope(np.array([5]), np.array([[1]]))
    Z_and = Z1 & Z2
    expected = Zonotope(np.array([4.5]), np.array([[0.5]]))
    assert Z_and.c is not None and expected.c is not None and np.allclose(Z_and.c, expected.c)
    assert Z_and.G is not None and expected.G is not None and np.allclose(Z_and.G, expected.G)

# 2D cases
def test_zonotope_and_2d_empty():
    """2D: intersection is empty set (MATLAB: Z1 & Z2; assert(representsa(Z_and,'emptySet')) )"""
    # Use deterministic generators to ensure no intersection
    Z1 = Zonotope(np.zeros(2), np.array([[1, 0], [0, 1]]))
    Z2 = Zonotope(5 * np.ones(2), np.array([[1, 0], [0, 1]]))
    Z_and = Z1 & Z2
    assert Z_and.representsa_('emptySet')

def test_zonotope_and_2d_point():
    """2D: intersection is a single point (MATLAB: Z1 & Z2; assert(~representsa(Z_and,'emptySet')) )"""
    Z1 = Zonotope(np.array([0, 0]), np.array([[1, 0.5], [0, 1]]))
    Z2 = Zonotope(np.array([2.5, 2.5]), np.array([[1, 0], [0.5, 1]]))
    Z_and = Z1 & Z2
    print('Z_and.c:', Z_and.c)
    print('Z_and.G:', Z_and.G)
    assert not Z_and.representsa_('emptySet')

def test_zonotope_and_2d_full():
    """2D: intersection is full-dimensional (MATLAB: Z1 & Z2; assert(~representsa(Z_and,'emptySet')) )"""
    # Use deterministic generators to ensure intersection
    Z1 = Zonotope(np.zeros(2), np.array([[1, 1], [0, 100]]))
    Z2 = Zonotope(np.zeros(2), np.array([[0.5, 0], [1, 0.5]]))
    Z_and = Z1 & Z2
    assert not Z_and.representsa_('emptySet')

def test_zonotope_and_empty_set():
    """Intersection with empty set (MATLAB: assert(representsa(Z1 & Z_e,'emptySet')) )"""
    Z1 = Zonotope(np.array([1, 2]), np.array([[1, -1, 2, 0], [1, 4, 0, 1]]))
    Z_e = EmptySet(2)
    assert (Z1.and_(Z_e)).representsa_('emptySet')

# Averaging method (all options)
def test_zonotope_and_averaging_method():
    """Averaging method (MATLAB: and(Z1,Z2,'averaging'); assert(~representsa(Z_and,'emptySet')) )"""
    Z1 = Zonotope(np.array([1, 2]), np.array([[1, -1, 2, 0], [1, 4, 0, 1]]))
    Z2 = Zonotope(np.array([-3, -4]), np.array([[1, 0, 2], [-1, 1, 2]]))
    Z_and = Z1.and_(Z2, 'averaging')
    assert not Z_and.representsa_('emptySet')

def test_zonotope_and_averaging_normgen_false():
    """Averaging method, normGen=False (MATLAB: and_(Z1,Z2,'averaging','normGen',false); assert(~representsa(Z_and,'emptySet')) )"""
    Z1 = Zonotope(np.array([1, 2]), np.array([[1, -1, 2, 0], [1, 4, 0, 1]]))
    Z2 = Zonotope(np.array([-3, -4]), np.array([[1, 0, 2], [-1, 1, 2]]))
    Z_and = Z1.and_(Z2, 'averaging', 'normGen', False)
    assert not Z_and.representsa_('emptySet')

def test_zonotope_and_averaging_normgen_true():
    """Averaging method, normGen=True (MATLAB: and_(Z1,Z2,'averaging','normGen',true,0.8); assert(~representsa(Z_and,'emptySet')) )"""
    Z1 = Zonotope(np.array([1, 2]), np.array([[1, -1, 2, 0], [1, 4, 0, 1]]))
    Z2 = Zonotope(np.array([-3, -4]), np.array([[1, 0, 2], [-1, 1, 2]]))
    Z_and = Z1.and_(Z2, 'averaging', 'normGen', True, 0.8)
    assert not Z_and.representsa_('emptySet')

def test_zonotope_and_averaging_radius():
    """Averaging method, radius (MATLAB: and_(Z1,Z2,'averaging','radius'); assert(~representsa(Z_and,'emptySet')) )"""
    Z1 = Zonotope(np.array([1, 2]), np.array([[1, -1, 2, 0], [1, 4, 0, 1]]))
    Z2 = Zonotope(np.array([-3, -4]), np.array([[1, 0, 2], [-1, 1, 2]]))
    Z_and = Z1.and_(Z2, 'averaging', 'radius')
    assert not Z_and.representsa_('emptySet')

def test_zonotope_and_averaging_volume():
    """Averaging method, volume (MATLAB: and_(Z1,Z2,'averaging','volume'); assert(~representsa(Z_and,'emptySet')) )"""
    Z1 = Zonotope(np.array([1, 2]), np.array([[1, -1, 2, 0], [1, 4, 0, 1]]))
    Z2 = Zonotope(np.array([-3, -4]), np.array([[1, 0, 2], [-1, 1, 2]]))
    Z_and = Z1.and_(Z2, 'averaging', 'volume')
    assert not Z_and.representsa_('emptySet')

if __name__ == "__main__":
    pytest.main([__file__]) 