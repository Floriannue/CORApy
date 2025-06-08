"""
test_linearSys_generateRandom - unit test for LinearSys.generateRandom

This test verifies that the generateRandom static method correctly creates
random linear systems with the specified properties.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

import pytest
import numpy as np
from cora_python.contDynamics import LinearSys


def test_generateRandom_default():
    """Test generateRandom with default parameters"""
    sys = LinearSys.generateRandom()
    
    # Check that system was created
    assert isinstance(sys, LinearSys)
    assert sys.nr_of_dims >= 4 and sys.nr_of_dims <= 10
    assert sys.nr_of_inputs >= 1 and sys.nr_of_inputs <= 3
    assert sys.nr_of_outputs >= 1 and sys.nr_of_outputs <= 2
    
    # Check matrix dimensions
    assert sys.A.shape == (sys.nr_of_dims, sys.nr_of_dims)
    assert sys.B.shape == (sys.nr_of_dims, sys.nr_of_inputs)
    assert sys.C.shape == (sys.nr_of_outputs, sys.nr_of_dims)


def test_generateRandom_specified_dimensions():
    """Test generateRandom with specified dimensions"""
    state_dim = 5
    input_dim = 2
    output_dim = 3
    
    sys = LinearSys.generateRandom(state_dimension=state_dim,
                                  input_dimension=input_dim,
                                  output_dimension=output_dim)
    
    # Check dimensions
    assert sys.nr_of_dims == state_dim
    assert sys.nr_of_inputs == input_dim
    assert sys.nr_of_outputs == output_dim
    
    # Check matrix dimensions
    assert sys.A.shape == (state_dim, state_dim)
    assert sys.B.shape == (state_dim, input_dim)
    assert sys.C.shape == (output_dim, state_dim)


def test_generateRandom_eigenvalue_intervals():
    """Test generateRandom with specified eigenvalue intervals"""
    state_dim = 4
    real_interval = (-2, -0.5)
    imaginary_interval = (-1, 1)
    
    sys = LinearSys.generateRandom(state_dimension=state_dim,
                                  real_interval=real_interval,
                                  imaginary_interval=imaginary_interval)
    
    # Check that system was created with correct dimensions
    assert sys.nr_of_dims == state_dim
    assert sys.A.shape == (state_dim, state_dim)
    
    # Check that eigenvalues are roughly in the specified range
    # (This is approximate due to numerical precision and transformation)
    eigenvals = np.linalg.eigvals(sys.A)
    real_parts = np.real(eigenvals)
    
    # Most eigenvalues should have real parts in the specified range
    # (allowing some tolerance for numerical precision)
    assert np.all(real_parts >= real_interval[0] - 0.1)
    assert np.all(real_parts <= real_interval[1] + 0.1)


def test_generateRandom_reproducibility():
    """Test that generateRandom produces different systems on different calls"""
    # Set seed for reproducibility in test
    np.random.seed(42)
    sys1 = LinearSys.generateRandom(state_dimension=3)
    
    np.random.seed(43)
    sys2 = LinearSys.generateRandom(state_dimension=3)
    
    # Systems should be different
    assert not np.allclose(sys1.A, sys2.A)
    assert not np.allclose(sys1.B, sys2.B)


def test_generateRandom_invalid_inputs():
    """Test generateRandom with invalid inputs"""
    
    # Test negative dimensions
    with pytest.raises(ValueError):
        LinearSys.generateRandom(state_dimension=-1)
    
    with pytest.raises(ValueError):
        LinearSys.generateRandom(input_dimension=0)
    
    with pytest.raises(ValueError):
        LinearSys.generateRandom(output_dimension=-5)


def test_generateRandom_edge_cases():
    """Test generateRandom with edge cases"""
    
    # Test minimal dimensions
    sys = LinearSys.generateRandom(state_dimension=1,
                                  input_dimension=1,
                                  output_dimension=1)
    
    assert sys.nr_of_dims == 1
    assert sys.nr_of_inputs == 1
    assert sys.nr_of_outputs == 1
    assert sys.A.shape == (1, 1)
    assert sys.B.shape == (1, 1)
    assert sys.C.shape == (1, 1)


if __name__ == "__main__":
    test_generateRandom_default()
    test_generateRandom_specified_dimensions()
    test_generateRandom_eigenvalue_intervals()
    test_generateRandom_reproducibility()
    test_generateRandom_invalid_inputs()
    test_generateRandom_edge_cases()
    print("All generateRandom tests passed!") 