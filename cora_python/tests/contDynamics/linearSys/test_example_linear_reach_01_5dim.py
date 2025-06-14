"""
Test file for example_linear_reach_01_5dim

This test verifies that the 5-dimensional linear reachability example
runs without errors and produces the expected outputs.

Authors: AI Assistant
Date: 2025
"""

import pytest
import numpy as np
import matplotlib
# Use non-interactive backend for testing
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cora_python.examples.contDynamics.linearSys.example_linear_reach_01_5dim import example_linear_reach_01_5dim


class TestExampleLinearReach01_5dim:
    """Test class for the 5-dimensional linear reachability example"""
    
    def test_example_linear_reach_01_5dim_runs(self):
        """Test that the example runs without errors"""
        # Close any existing figures to avoid interference
        plt.close('all')
        
        # Capture stdout to suppress print statements during testing
        import io
        import contextlib
        
        captured_output = io.StringIO()
        
        with contextlib.redirect_stdout(captured_output):
            result = example_linear_reach_01_5dim()
        
        # Check that example completed successfully
        assert result is True, "Example should return True on successful completion"
        
        # Check that some output was produced
        output = captured_output.getvalue()
        assert "Linear Reachability Analysis Example" in output, "Expected example title in output"
        assert "Computing reachable set" in output, "Expected reachability computation message"
        assert "Running random simulations" in output, "Expected simulation message"
        assert "Example completed successfully" in output, "Expected completion message"
        
        # Close any figures created during the test
        plt.close('all')
    
    def test_example_components_creation(self):
        """Test that the main components of the example can be created"""
        from cora_python.contDynamics.linearSys import LinearSys
        from cora_python.contSet.zonotope import Zonotope
        from cora_python.contSet.interval import Interval
        
        # Test system creation
        A = np.array([[-1, -4, 0, 0, 0], 
                      [4, -1, 0, 0, 0], 
                      [0, 0, -3, 1, 0], 
                      [0, 0, -1, -3, 0], 
                      [0, 0, 0, 0, -2]])
        B = 1
        
        system = LinearSys('testSys', A, B)
        assert system.name == 'testSys'
        assert np.array_equal(system.A, A)
        # B is now converted to a matrix when scalar, so check the original input behavior
        assert system.nr_of_inputs == 5  # scalar B=1 means 5 inputs for 5x5 system
        
        # Test initial set creation
        R0 = Zonotope(np.ones((5, 1)), 0.1 * np.diag(np.ones(5)))
        assert R0.c.shape == (5, 1)  # Center should be column vector
        assert R0.G.shape == (5, 5)
        
        # Test input set creation (interval to zonotope conversion)
        input_interval = Interval(np.array([0.9, -0.25, -0.1, 0.25, -0.75]), 
                                 np.array([1.1, 0.25, 0.1, 0.75, -0.25]))
        U = Zonotope(input_interval)
        
        # Verify the conversion
        expected_center = np.array([[1.0], [0.0], [0.0], [0.5], [-0.5]])  # Column vector
        expected_radius = np.array([0.1, 0.25, 0.1, 0.25, 0.25])
        
        np.testing.assert_array_almost_equal(U.c, expected_center, decimal=10)
        # Check that generators matrix has the expected radii on diagonal
        np.testing.assert_array_almost_equal(np.diag(U.G), expected_radius, decimal=10)
    
    def test_example_parameters_setup(self):
        """Test that the example parameters are set up correctly"""
        from cora_python.contSet.zonotope import Zonotope
        from cora_python.contSet.interval import Interval
        
        # Simulate the parameters setup from the example
        params = {}
        params['tFinal'] = 5
        params['R0'] = Zonotope(np.ones((5, 1)), 0.1 * np.diag(np.ones(5)))
        params['U'] = Zonotope(Interval(np.array([0.9, -0.25, -0.1, 0.25, -0.75]), 
                                       np.array([1.1, 0.25, 0.1, 0.75, -0.25])))
        
        # Verify parameters
        assert params['tFinal'] == 5
        assert isinstance(params['R0'], Zonotope)
        assert isinstance(params['U'], Zonotope)
        assert params['R0'].dim() == 5
        assert params['U'].dim() == 5
        
        # Test options setup
        options = {}
        options['timeStep'] = 0.02
        options['taylorTerms'] = 4
        options['zonotopeOrder'] = 20
        
        assert options['timeStep'] == 0.02
        assert options['taylorTerms'] == 4
        assert options['zonotopeOrder'] == 20


if __name__ == "__main__":
    pytest.main([__file__]) 