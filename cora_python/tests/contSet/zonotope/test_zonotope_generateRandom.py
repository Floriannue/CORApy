"""
test_zonotope_generateRandom - unit test function of generateRandom

Syntax:
    pytest test_zonotope_generateRandom.py

Inputs:
    -

Outputs:
    test results

Other files required: none
Subfunctions: none
Files required: none

See also: -

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       27-September-2019 (MATLAB)
Last update:   19-May-2022 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope


def test_zonotope_generateRandom_empty_call():
    """Test empty call to generateRandom"""
    Z = Zonotope.generateRandom()
    assert isinstance(Z, Zonotope)
    assert Z.c is not None
    assert Z.G is not None


def test_zonotope_generateRandom_dimension_only():
    """Test generateRandom with only dimension parameter"""
    n = 3
    Z = Zonotope.generateRandom('Dimension', n)
    assert Z.dim() == n


def test_zonotope_generateRandom_center_only():
    """Test generateRandom with only center parameter"""
    c = np.array([[2], [1], [-1]])
    Z = Zonotope.generateRandom('Center', c)
    assert np.allclose(Z.c, c)


def test_zonotope_generateRandom_nr_generators_only():
    """Test generateRandom with only number of generators parameter"""
    nrGens = 10
    Z = Zonotope.generateRandom('NrGenerators', nrGens)
    assert Z.G.shape[1] == nrGens


def test_zonotope_generateRandom_distribution_only():
    """Test generateRandom with only distribution parameter"""
    type_dist = 'exp'
    Z = Zonotope.generateRandom('Distribution', type_dist)
    assert isinstance(Z, Zonotope)


def test_zonotope_generateRandom_dimension_and_nr_generators():
    """Test generateRandom with dimension and number of generators"""
    n = 3
    nrGens = 10
    Z = Zonotope.generateRandom('Dimension', n, 'NrGenerators', nrGens)
    assert Z.dim() == n and Z.G.shape[1] == nrGens


def test_zonotope_generateRandom_center_and_nr_generators():
    """Test generateRandom with center and number of generators"""
    c = np.array([[2], [1], [-1]])
    nrGens = 10
    Z = Zonotope.generateRandom('Center', c, 'NrGenerators', nrGens)
    assert np.allclose(Z.c, c) and Z.G.shape[1] == nrGens


def test_zonotope_generateRandom_center_and_distribution():
    """Test generateRandom with center and distribution"""
    c = np.array([[2], [1], [-1]])
    type_dist = 'exp'
    Z = Zonotope.generateRandom('Center', c, 'Distribution', type_dist)
    assert np.allclose(Z.c, c)


def test_zonotope_generateRandom_nr_generators_and_distribution():
    """Test generateRandom with number of generators and distribution"""
    nrGens = 10
    type_dist = 'exp'
    Z = Zonotope.generateRandom('NrGenerators', nrGens, 'Distribution', type_dist)
    assert Z.G.shape[1] == nrGens


def test_zonotope_generateRandom_all_parameters():
    """Test generateRandom with all parameters"""
    c = np.array([[2], [1], [-1]])
    nrGens = 10
    type_dist = 'exp'
    Z = Zonotope.generateRandom('Center', c, 'NrGenerators', nrGens, 'Distribution', type_dist)
    assert np.allclose(Z.c, c) and Z.G.shape[1] == nrGens


def test_zonotope_generateRandom_uniform_distribution():
    """Test generateRandom with uniform distribution"""
    Z = Zonotope.generateRandom('Distribution', 'uniform')
    assert isinstance(Z, Zonotope)


def test_zonotope_generateRandom_gamma_distribution():
    """Test generateRandom with gamma distribution"""
    Z = Zonotope.generateRandom('Distribution', 'gamma')
    assert isinstance(Z, Zonotope)


def test_zonotope_generateRandom_center_from_dimension():
    """Test that center dimension is inferred from center parameter"""
    c = np.array([[1], [2], [3], [4]])
    Z = Zonotope.generateRandom('Center', c)
    assert Z.dim() == 4
    assert np.allclose(Z.c, c)


def test_zonotope_generateRandom_default_generators():
    """Test that default number of generators is 2*dimension"""
    n = 5
    Z = Zonotope.generateRandom('Dimension', n)
    assert Z.G.shape[1] == 2 * n


def test_zonotope_generateRandom_center_vector_input():
    """Test generateRandom with center as a vector (not column vector)"""
    c = np.array([2, 1, -1])
    Z = Zonotope.generateRandom('Center', c)
    assert np.allclose(Z.c.flatten(), c)


def test_zonotope_generateRandom_center_list_input():
    """Test generateRandom with center as a list"""
    c = [2, 1, -1]
    Z = Zonotope.generateRandom('Center', c)
    assert np.allclose(Z.c.flatten(), np.array(c))


if __name__ == "__main__":
    pytest.main([__file__]) 