import numpy as np
import pytest
from cora_python.g.classes.testCase.testCase import CoraTestCase

@pytest.fixture
def base_test_case():
    """Provides a basic TestCase instance for testing."""
    y = np.arange(10).reshape(5, 2, 1)
    u = np.arange(5).reshape(5, 1, 1)
    x = y + 1
    dt = 0.5
    return CoraTestCase(y, u, x, dt, "Test1")

def test_testCase_init(base_test_case):
    """Test constructor and basic properties."""
    assert base_test_case.y.shape == (5, 2, 1)
    assert base_test_case.u.shape == (5, 1, 1)
    assert base_test_case.x.shape == (5, 2, 1)
    assert base_test_case.sampleTime == 0.5
    assert base_test_case.name == "Test1"
    
    # Check initial state extraction - use the same logic as the constructor
    initial_slice = base_test_case.x[0:1, :, :]
    expected_initialState = np.transpose(initial_slice, (1, 0, 2))
    np.testing.assert_array_equal(base_test_case.initialState, expected_initialState)

def test_set_u(base_test_case):
    """Test the set_u method."""
    new_u = np.ones((5, 1, 1))
    base_test_case.set_u(new_u)
    np.testing.assert_array_equal(base_test_case.u, new_u)

def test_reduceLength(base_test_case):
    """Test the reduceLength method."""
    original_len = base_test_case.y.shape[0]
    new_len = 3
    base_test_case.reduceLength(new_len)
    
    assert base_test_case.y.shape[0] == new_len
    assert base_test_case.u.shape[0] == new_len
    # The 'x' state matrix is not reduced in the MATLAB version, so we don't check it.

def test_combineTestCases():
    """Test the combineTestCases method."""
    y1 = np.ones((3, 2, 1))
    u1 = np.ones((3, 1, 1))
    x1 = np.ones((3, 2, 1))
    tc1 = CoraTestCase(y1, u1, x1, 0.1)
    
    y2 = np.zeros((3, 2, 2)) # 2 trajectories
    u2 = np.zeros((3, 1, 2))
    x2 = np.zeros((3, 2, 2))
    tc2 = CoraTestCase(y2, u2, x2, 0.1)

    combined_tc = tc1.combineTestCases(tc2)

    # Check initial state stacking (3 trajectories total)
    assert combined_tc.initialState.shape == (2, 1, 3)
    # Check u stacking
    assert combined_tc.u.shape == (3, 1, 3)
    # Check y stacking
    assert combined_tc.y.shape == (3, 2, 3)
    
    # Check values
    np.testing.assert_array_equal(combined_tc.y[:,:,0], y1[:,:,0])
    np.testing.assert_array_equal(combined_tc.y[:,:,1:], y2)

def test_init_with_list_y():
    """Test constructor with y as a list of arrays."""
    y_list = [np.random.rand(5, 2) for _ in range(3)]
    u = np.random.rand(5, 1, 3)
    x = np.random.rand(5, 2, 3)
    tc = CoraTestCase(y_list, u, x, 0.1)
    
    assert tc.y.shape == (5, 2, 3)
    np.testing.assert_array_equal(tc.y[:,:,0], y_list[0])

# --- Mocks for Advanced Method Testing ---

class MockSimulateSys:
    """A mock system with a 'simulate' method for testing compute_ya."""
    def __init__(self, dt=0.1, y_dim=2):
        self.dt = dt
        self.y_dim = y_dim

    def simulate(self, params):
        # Pretend the nominal output is just half of the input 'u'
        # and doesn't depend on x0 or tFinal
        u = params['u']
        # The nominal output should have the same dimensions as y
        y_nom_s = np.ones((u.shape[1], self.y_dim)) * (u.T / 2)
        return None, None, None, y_nom_s

def test_compute_ya(base_test_case):
    """Test the computation of measurement deviation y_a."""
    mock_sys = MockSimulateSys(dt=base_test_case.sampleTime, y_dim=base_test_case.y.shape[1])
    
    # Make u simple to calculate expected y_a
    base_test_case.u = np.full((5, 1, 1), 10.0)
    
    # compute_ya modifies the object in place
    base_test_case.compute_ya(mock_sys)
    
    expected_y_nom = np.full((5, 1, 1), 5.0)
    # y in base_test_case is arange(10).reshape(5,2,1)
    # The mock system only produces 1-dim output, so y_a will broadcast
    expected_ya = base_test_case.y - expected_y_nom

    assert base_test_case.y_a is not None
    np.testing.assert_allclose(base_test_case.y_a, expected_ya)

def test_setInitialStateToMeas(base_test_case):
    """Test setting the initial state from the first p measurements."""
    # Case 1: All trajectories start with the same measurements
    y = np.random.rand(10, 2, 5)
    y[:2, :, :] = np.random.rand(2, 2, 1) # Make first 2 steps identical for all 5 traj
    x = np.copy(y)
    tc = CoraTestCase(y, np.zeros_like(y), x, 0.1)
    
    res = tc.setInitialStateToMeas(p=2)
    assert len(res) == 1
    expected_initialState = y[:2,:,0].T.flatten()
    np.testing.assert_allclose(res[0].initialState, expected_initialState)

    # Case 2: Trajectories have different starting measurements
    y = np.random.rand(10, 2, 3) # 3 different trajectories
    x = np.copy(y)
    tc_diff = CoraTestCase(y, np.zeros_like(y), x, 0.1)

    res_diff = tc_diff.setInitialStateToMeas(p=2)
    assert len(res_diff) == 3
    # Check the second test case's initial state
    expected_is2 = y[:2,:,1].T.flatten()
    np.testing.assert_allclose(res_diff[1].initialState, expected_is2)
    assert res_diff[1].y.shape == y[:,:,1].shape

def test_to_simResult(base_test_case):
    """Test the conversion to a simResult object."""
    # Need to mock or import the actual simResult class
    from cora_python.g.classes.simResult.simResult import SimResult
    
    simRes = base_test_case.to_simResult()
    assert isinstance(simRes, SimResult)

    # Check content
    nrOfTimeSteps = base_test_case.y.shape[0]
    expected_tVec = np.arange(0, base_test_case.sampleTime * nrOfTimeSteps, base_test_case.sampleTime)
    
    assert len(simRes.x) == 1
    np.testing.assert_allclose(simRes.x[0], base_test_case.x)
    assert len(simRes.t) == 1
    np.testing.assert_allclose(simRes.t[0], expected_tVec)
    assert len(simRes.y) == 1
    np.testing.assert_allclose(simRes.y[0], base_test_case.y)

def test_plot(base_test_case, mocker):
    """Test that the plot method calls the simResult plot method."""
    # Mock the plot method of the simResult object that gets created
    mock_sim_plot = mocker.patch('cora_python.g.classes.simResult.simResult.SimResult.plot')
    
    base_test_case.plot([0, 1], Traj='y')

    # Assert that the mocked plot method was called
    mock_sim_plot.assert_called_once()
    # Assert it was called with the correct arguments
    args, kwargs = mock_sim_plot.call_args
    assert args == ([0, 1],)
    assert kwargs == {'Traj': 'y'} 