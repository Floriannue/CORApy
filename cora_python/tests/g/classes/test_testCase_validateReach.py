import numpy as np
import pytest
from cora_python.g.classes.testCase.testCase import CoraTestCase
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contDynamics.linearARX import linearARX
from cora_python.contDynamics.nonlinearARX import nonlinearARX

# --- Mock Objects for Testing ---

class MockReachSet:
    def __init__(self, sets):
        self.timePoint = self
        self.set = sets

    def __init__(self, dim=2):
        self.dt = 0.1
        self.nrOfInputs = 1
        self.dim = dim

class MockSys:
    def __init__(self, dim=2):
        self.dt = 0.1
        self.nrOfInputs = 1
        self.dim = dim
    
    def reach(self, params, options):
        # Return a reachable set that always contains the measurements
        n_k = params['u'].shape[1]
        # Create a large zonotope that will contain anything
        z = Zonotope(np.zeros((self.dim,1)), np.eye(self.dim) * 1000)
        return MockReachSet([z] * n_k)

    def simulate(self, params):
        # Mock simulation returns a zero trajectory of the correct size
        t_final = params.get('tFinal', 0.9)
        num_steps = int(round(t_final / self.dt)) + 1
        y_nom_s = np.zeros((num_steps, self.dim))
        # Return four values to match the expected output of a real simulate call
        return None, None, None, y_nom_s

@pytest.fixture
def sample_configs(validation_test_case):
    """Create a sample configs structure for testing."""
    dim = validation_test_case.y.shape[1]
    sys = MockSys(dim=dim)
    params = {
        'R0': Zonotope(np.zeros((dim,1)), np.eye(dim)*0.1),
        'U': Zonotope(np.array([[0]]), np.array([[0.05]])),
    }
    options = {'zonotopeOrder': 50}
    
    configs = [
        {'sys': sys, 'params': params, 'options': options, 'name': 'true'},
        {'sys': sys, 'params': params, 'options': options, 'name': 'different'}
    ]
    return configs

@pytest.fixture
def validation_test_case():
    """Create a TestCase object for validation."""
    y = np.random.rand(10, 2, 5) # 10 steps, 2 dims, 5 trajectories
    u = np.random.rand(10, 1, 5)
    # For this test, x can be the same as y for simplicity
    x = np.copy(y)
    dt = 0.1
    return CoraTestCase(y, u, x, dt)

# --- Tests ---

def test_validateReach_computation(sample_configs):
    """Test that validateReach runs without errors and returns values."""
    test_case = sample_configs['test_case']
    configs = sample_configs['configs']
    
    res = test_case.validateReach(configs)
    
    assert 'res' in res
    assert 'R' in res
    assert 'R_p' in res
    assert 'plot_h' in res
    assert len(res['res']) == len(configs)

def test_validateReach_containment(mocker):
    """Test the containment check logic in validateReach."""
    
    # Create a simple test case with one trajectory
    y = np.array([[[0.5], [0.5]], [[0.6], [0.6]]]) # T=2, dim=2, traj=1
    u = np.zeros((2, 1, 1))
    test_case = CoraTestCase(y, u, y, 0.1)

    # Mock the reachability analysis to return a known set
    class SmallReachSys:
        def reach(self, params, options):
            # A small zonotope around the origin
            r_set = Zonotope(np.zeros((2,1)), np.eye(2) * 0.1)
            # Must return a list of sets for each time step
            return [r_set] * 2
        
        def simulate(self, params):
            return None, None, None, np.zeros((2, 2))

    configs = [{
        'name': 'TestSys',
        'sys': SmallReachSys(),
        'obj': test_case,
        'params': {'R0': Zonotope(np.zeros((2,1)), np.eye(2) * 0.01)},
        'options': {}
    }]

    # With the small reachable set, the measurement y should NOT be contained.
    # We expect a warning.
    mocker.patch('warnings.warn')
    test_case.validateReach(configs)
    warnings.warn.assert_called()

    # Now, create a system that produces a huge reachable set
    class BigReachSys:
        def reach(self, params, options):
            r_set = Zonotope(np.zeros((2,1)), np.eye(2) * 1000)
            return [r_set] * 2

        def simulate(self, params):
            return None, None, None, np.zeros((2, 2))
            
    configs[0]['sys'] = BigReachSys()
    
    # Reset mock and run again. This time, no warning should be issued.
    warnings.warn.reset_mock()
    test_case.validateReach(configs)
    warnings.warn.assert_not_called() 