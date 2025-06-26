import numpy as np
import pytest
from cora_python.g.classes.testCase.testCase import CoraTestCase

@pytest.fixture
def sample_test_case():
    """Create a sample TestCase object for testing."""
    n_samples = 50
    n_states = 3
    n_inputs = 2
    
    # Create sample data
    y = np.random.rand(n_samples, n_states, 1)
    u = np.random.rand(n_samples, n_inputs, 1)
    x = np.random.rand(n_samples, n_states, 1)
    dt = 0.1
    
    return CoraTestCase(y, u, x, dt)

def test_sequentialTestCases(sample_test_case):
    """
    Test the sequentialTestCases method.
    This test replicates the logic from test_testCase_sequentialTestCases.m.
    """
    length = 20
    seqTestCases = sample_test_case.sequentialTestCases(length)

    # Check number of generated test cases
    assert len(seqTestCases) == sample_test_case.y.shape[0] - length

    # Check correctness of each test case
    for i in range(len(seqTestCases) - 1):
        tc1 = seqTestCases[i]
        tc2 = seqTestCases[i+1]
        
        # Check if values starting from time step 2 equal first values of next test case
        np.testing.assert_allclose(tc1.y[1:, :], tc2.y[:-1, :])
        np.testing.assert_allclose(tc1.u[1:, :], tc2.u[:-1, :])
        np.testing.assert_allclose(tc1.x[1:, :], tc2.x[:-1, :])
        
        # Check initial state
        np.testing.assert_allclose(tc1.x[1, :], tc2.initialState.T)

def test_sequentialTestCases_edge_cases(sample_test_case):
    """Test edge cases for sequentialTestCases."""
    # Length equals total samples, should return one test case (the object itself)
    res = sample_test_case.sequentialTestCases(sample_test_case.y.shape[0])
    assert len(res) == 1
    assert res[0] is sample_test_case

    # Length greater than total samples, should also return one
    res = sample_test_case.sequentialTestCases(sample_test_case.y.shape[0] + 1)
    assert len(res) == 1
    assert res[0] is sample_test_case
    
    # No states provided
    sample_test_case.x = None
    with pytest.raises(ValueError, match="state must be available"):
        sample_test_case.sequentialTestCases(10) 