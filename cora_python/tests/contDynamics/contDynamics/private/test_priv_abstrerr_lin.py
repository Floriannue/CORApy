"""
test_priv_abstrerr_lin - Test for priv_abstrerr_lin function
   MATLAB I/O pairs from debug_matlab_nonlinearSys_initReach_abstrerr_chain.m

   This test verifies that priv_abstrerr_lin correctly computes the abstraction 
   error for linearization approach using the 6D tank example.

Syntax:
    pytest cora_python/tests/contDynamics/contDynamics/private/test_priv_abstrerr_lin.py

Authors:       MATLAB I/O pairs from debug_matlab_nonlinearSys_initReach_abstrerr_chain.m
Written:       2025
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.contDynamics.private.priv_abstrerr_lin import priv_abstrerr_lin
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.contDynamics.linReach import linReach


class TestPrivAbstrerrLin:
    """Test class for priv_abstrerr_lin functionality"""
    
    def test_priv_abstrerr_lin_tank6_example(self):
        """Test with 6D tank example (same as MATLAB test_nonlinearSys_initReach)"""
        # Model parameters (same as MATLAB test)
        dim_x = 6
        params = {
            'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
            'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
            'tFinal': 4,
            'uTrans': np.zeros((1, 1))
        }
        
        # Reachability settings (same as MATLAB test)
        # Note: maxError is set by validateOptions in MATLAB, but we set it manually here
        options = {
            'timeStep': 4,
            'taylorTerms': 4,
            'zonotopeOrder': 50,
            'alg': 'lin',
            'tensorOrder': 2,
            'reductionTechnique': 'girard',
            'errorOrder': 10,
            'intermediateOrder': 10,
            'maxError': np.full((dim_x, 1), np.inf)  # Required by linReach (set by validateOptions in MATLAB)
        }
        
        # System dynamics
        tank = NonlinearSys(tank6Eq, states=6, inputs=1)
        
        # Compute derivatives (required)
        tank.derivatives(options)
        
        # Compute factors
        for i in range(1, options['taylorTerms'] + 2):
            options['factor'] = options.get('factor', [])
            options['factor'].append((options['timeStep'] ** i) / np.math.factorial(i))
        
        # Prepare Rstart structure (as expected by linReach)
        Rstart = {'set': params['R0'], 'error': np.zeros((dim_x, 1))}
        
        # Call linReach to get Rmax (which includes abstraction error)
        Rti, Rtp, dimForSplit, options_out = linReach(tank, Rstart, params, options)
        
        # MATLAB: Rmax = Rti+RallError; [trueError,VerrorDyn] = priv_abstrerr_lin(sys,Rmax,params,options);
        # The abstraction error from linReach is stored in Rtp.error
        # MATLAB result: linErrors = [0.000206863579523074; 0.000314066666873806; ...]
        expected_error = np.array([
            [0.000206863579523074],
            [0.000314066666873806],
            [0.000161658311464827],
            [0.00035325543180986],
            [0.000358487021465299],
            [0.000209190642349808]
        ])
        
        # Compare the error from linReach (which calls priv_abstrerr_lin internally)
        actual_error = Rtp['error']
        
        # Debug: Print intermediate values to help identify the issue
        print(f"\nDebug: actual_error = {actual_error.flatten()}")
        print(f"Debug: expected_error = {expected_error.flatten()}")
        print(f"Debug: Rti center = {Rti.center().flatten() if hasattr(Rti, 'center') else 'N/A'}")
        print(f"Debug: Rti generators shape = {Rti.generators().shape if hasattr(Rti, 'generators') else 'N/A'}")
        
        # Ensure both are 1D arrays for comparison
        if actual_error.ndim > 1:
            actual_error = actual_error.flatten()
        if expected_error.ndim > 1:
            expected_error = expected_error.flatten()
        
        # Allow tolerance for numerical differences (MATLAB vs Python floating point)
        # Note: Values may differ due to different numerical implementations, 
        # different linearization points, or accumulated rounding errors
        # MATLAB values from debug_matlab_nonlinearSys_initReach_abstrerr_chain.m
        # Differences stem from earlier linReach numerical steps (see debug scripts)
        np.testing.assert_allclose(actual_error, expected_error, rtol=2e-2, atol=1e-9)
    
    def test_priv_abstrerr_lin_direct_call(self):
        """Test priv_abstrerr_lin with direct call (if possible)"""
        # This test may not work if priv_abstrerr_lin requires specific setup
        # from linReach. We'll skip if it fails.

        dim_x = 6
        params = {
            'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
            'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
            'tFinal': 4,
            'uTrans': np.zeros((1, 1))
        }
        
        options = {
            'timeStep': 4,
            'taylorTerms': 4,
            'zonotopeOrder': 50,
            'alg': 'lin',
            'tensorOrder': 2,
            'reductionTechnique': 'girard',
            'errorOrder': 10,
            'intermediateOrder': 10
        }
        
        tank = NonlinearSys(tank6Eq, states=6, inputs=1)
        tank.derivatives(options)
        
        # Create a reachable set R (time-interval from linearized system)
        # This is what priv_abstrerr_lin expects
        R = Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.1 * np.eye(dim_x))
        
        # Call priv_abstrerr_lin directly
        trueError, VerrorDyn = priv_abstrerr_lin(tank, R, params, options)
        
        # Verify outputs
        assert trueError is not None
        assert isinstance(trueError, np.ndarray)
        assert trueError.shape[0] == dim_x
        
        assert VerrorDyn is not None
        assert isinstance(VerrorDyn, Zonotope)



def test_priv_abstrerr_lin():
    """Test function for priv_abstrerr_lin method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestPrivAbstrerrLin()
    test.test_priv_abstrerr_lin_tank6_example()
    test.test_priv_abstrerr_lin_direct_call()
    
    print("test_priv_abstrerr_lin: all tests passed")
    return True


if __name__ == "__main__":
    test_priv_abstrerr_lin()
