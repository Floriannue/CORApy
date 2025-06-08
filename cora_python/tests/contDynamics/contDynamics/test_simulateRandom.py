"""
test_simulateRandom - unit test for simulateRandom function

This test verifies the functionality of the simulateRandom function for
contDynamics objects, specifically testing the linearSys implementation.

Authors: Florian Nüssel (Python implementation)
Date: 2025-01-08
"""

import pytest
import numpy as np
from cora_python.contDynamics.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.g.classes import SimResult


class TestSimulateRandom:
    """Test class for simulateRandom function"""
    
    def test_basic_simulation(self):
        """Test basic random simulation functionality"""
        # System dynamics
        A = np.array([[-0.7, -2], [2, -0.7]])
        B = np.array([[1], [1]])
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tFinal': 1.0,
            'R0': Zonotope(np.array([[2], [2]]), 0.1 * np.eye(2)),
            'U': Zonotope(np.array([[0]]), np.array([[0.1]]))
        }
        
        # Simulation settings
        options = {
            'points': 3,
            'fracVert': 0.5,
            'fracInpVert': 0.5,
            'nrConstInp': 1
        }
        
        # Random simulation
        simRes = sys.simulateRandom(params, options)
        
        # Verify results
        assert isinstance(simRes, list)
        assert len(simRes) == options['points']
        
        for sim in simRes:
            assert isinstance(sim, SimResult)
            assert len(sim.x) == 1  # One trajectory per simulation
            assert len(sim.t) == 1  # One time vector per simulation
            assert sim.x[0].shape[1] == sys.nr_of_dims  # Correct state dimension
            assert sim.t[0][0] == params.get('tStart', 0)  # Correct start time
            assert abs(sim.t[0][-1] - params['tFinal']) < 1e-10  # Correct end time
    
    def test_multiple_input_segments(self):
        """Test simulation with multiple piecewise-constant input segments"""
        # System dynamics
        A = np.array([[-1, 0], [0, -2]])
        B = np.array([[1], [1]])
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tFinal': 2.0,
            'R0': Zonotope(np.array([[1], [1]]), 0.05 * np.eye(2)),
            'U': Zonotope(np.array([[0]]), np.array([[0.1]]))
        }
        
        # Simulation settings with multiple input segments
        options = {
            'points': 2,
            'fracVert': 1.0,  # All extreme points
            'fracInpVert': 1.0,  # All extreme inputs
            'nrConstInp': 5  # 5 input segments
        }
        
        # Random simulation
        simRes = sys.simulateRandom(params, options)
        
        # Verify results
        assert len(simRes) == options['points']
        
        for sim in simRes:
            assert isinstance(sim, SimResult)
            assert len(sim.x) == 1
            assert len(sim.t) == 1
            assert sim.x[0].shape[1] == sys.nr_of_dims
    
    def test_with_output_equation(self):
        """Test simulation with output equation"""
        # System dynamics with output
        A = np.array([[-1, -2], [1, -1]])
        B = np.array([[1], [0]])
        C = np.array([[1, 0], [0, 1]])  # Identity output
        sys = LinearSys(A, B, C=C)
        
        # Parameters
        params = {
            'tFinal': 1.0,
            'R0': Zonotope(np.array([[1], [0]]), 0.1 * np.eye(2)),
            'U': Zonotope(np.array([[0]]), np.array([[0.05]]))
        }
        
        # Simulation settings
        options = {
            'points': 2,
            'fracVert': 0.5,
            'fracInpVert': 0.5,
            'nrConstInp': 1
        }
        
        # Random simulation
        simRes = sys.simulateRandom(params, options)
        
        # Verify results
        assert len(simRes) == options['points']
        
        for sim in simRes:
            assert isinstance(sim, SimResult)
            assert len(sim.x) == 1
            assert len(sim.t) == 1
            assert len(sim.y) == 1  # Output should be present
            assert sim.y[0].shape[1] == sys.nr_of_outputs
    
    def test_interval_initial_set(self):
        """Test simulation with interval initial set"""
        # System dynamics
        A = np.array([[-0.5, 0], [0, -1]])
        B = np.array([[1], [1]])
        sys = LinearSys(A, B)
        
        # Parameters with interval initial set
        params = {
            'tFinal': 1.0,
            'R0': Interval(np.array([0.9, -0.1]), np.array([1.1, 0.1])),
            'U': Zonotope(np.array([[0]]), np.array([[0.1]]))
        }
        
        # Simulation settings
        options = {
            'points': 3,
            'fracVert': 0.5,
            'fracInpVert': 0.5,
            'nrConstInp': 1
        }
        
        # Random simulation
        simRes = sys.simulateRandom(params, options)
        
        # Verify results
        assert len(simRes) == options['points']
        
        for sim in simRes:
            assert isinstance(sim, SimResult)
            assert len(sim.x) == 1
            assert len(sim.t) == 1
            assert sim.x[0].shape[1] == sys.nr_of_dims
    
    def test_default_options(self):
        """Test simulation with default options"""
        # System dynamics
        A = np.array([[-1]])
        B = np.array([[1]])
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tFinal': 0.5,
            'R0': Zonotope(np.array([[1]]), np.array([[0.1]]))
        }
        
        # Random simulation with default options
        simRes = sys.simulateRandom(params)
        
        # Verify results
        assert isinstance(simRes, list)
        assert len(simRes) == 1  # Default points = 1
        
        sim = simRes[0]
        assert isinstance(sim, SimResult)
        assert len(sim.x) == 1
        assert len(sim.t) == 1
        assert sim.x[0].shape[1] == sys.nr_of_dims
    
    def test_empty_input_set(self):
        """Test simulation without explicit input set"""
        # System dynamics
        A = np.array([[-2, 1], [0, -1]])
        B = np.array([[1], [1]])
        sys = LinearSys(A, B)
        
        # Parameters without input set
        params = {
            'tFinal': 1.0,
            'R0': Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2))
        }
        
        # Simulation settings
        options = {
            'points': 2,
            'fracVert': 0.5,
            'fracInpVert': 0.5,
            'nrConstInp': 1
        }
        
        # Random simulation
        simRes = sys.simulateRandom(params, options)
        
        # Verify results
        assert len(simRes) == options['points']
        
        for sim in simRes:
            assert isinstance(sim, SimResult)
            assert len(sim.x) == 1
            assert len(sim.t) == 1
    
    def test_time_consistency(self):
        """Test that simulation times are consistent"""
        # System dynamics
        A = np.array([[-1, 0], [0, -2]])
        B = np.array([[1], [1]])
        sys = LinearSys(A, B)
        
        # Parameters
        params = {
            'tStart': 0.5,
            'tFinal': 2.0,
            'R0': Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2)),
            'U': Zonotope(np.array([[0]]), np.array([[0.1]]))
        }
        
        # Simulation settings
        options = {
            'points': 2,
            'fracVert': 0.5,
            'fracInpVert': 0.5,
            'nrConstInp': 1
        }
        
        # Random simulation
        simRes = sys.simulateRandom(params, options)
        
        # Verify time consistency
        for sim in simRes:
            t_vec = sim.t[0]
            assert abs(t_vec[0] - params['tStart']) < 1e-10
            assert abs(t_vec[-1] - params['tFinal']) < 1e-10
            # Time should be monotonically increasing
            assert np.all(np.diff(t_vec) >= 0)


if __name__ == '__main__':
    pytest.main([__file__]) 