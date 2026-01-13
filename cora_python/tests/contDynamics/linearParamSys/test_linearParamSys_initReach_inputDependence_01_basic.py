"""
Test file for initReach_inputDependence - translated from MATLAB

This test verifies the initReach_inputDependence function for linearParamSys.

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contDynamics.linearParamSys import LinearParamSys
from cora_python.contDynamics.linearParamSys.initReach_inputDependence import initReach_inputDependence
from cora_python.contSet.zonotope import Zonotope
# center and generators are attached to Zonotope class, use object.center() and object.generators()


class TestInitReachInputDependence:
    """Test cases for initReach_inputDependence"""
    
    def test_initReach_inputDependence_01_simple_2d_system(self):
        """Test Case 1: Simple 2D system with constant parameters"""
        # MATLAB I/O pairs from debug_matlab_initReach_inputDependence.m
        # TODO: Add MATLAB I/O pairs after running debug script
        
        # Setup - A must be matZonotope or intervalMatrix for expmMixed
        # Use 2 generators to avoid expmOneParam path (which requires linearized params)
        from cora_python.matrixSet.matZonotope import matZonotope
        A_center = np.array([[0, 1], [-1, -0.5]])
        A_gen = np.zeros((2, 2, 2))
        A_gen[:, :, 0] = np.array([[0.1, 0], [0, 0.1]])  # First generator
        A_gen[:, :, 1] = np.array([[0, 0.05], [0.05, 0]])  # Second generator
        A = matZonotope(A_center, A_gen)
        B = np.array([[0], [1]])
        c = np.array([[0], [0]])
        sys = LinearParamSys(A, B, c, 'constParam')
        
        # Initial set
        Rinit = Zonotope(np.array([[0], [0]]), np.array([[0.1, 0], [0, 0.1]]))
        
        # Parameters
        # U is in input dimension (1D), Uconst is in input dimension (1D), uTrans is in input dimension (1D)
        # After linearization, these get transformed to state space dimension
        params = {
            'U': Zonotope(np.array([[0]]), np.array([[0.05]])),  # Input dimension (1D)
            'Uconst': Zonotope(np.array([[0]]), np.array([[0.05]])),  # Input dimension (1D)
            'uTrans': np.array([[0.1]])  # Input dimension (1D)
        }
        
        # Options
        options = {
            'timeStep': 0.1,
            'taylorTerms': 4,
            'reductionTechnique': 'girard',
            'zonotopeOrder': 10,
            'compTimePoint': True,
            'intermediateTerms': 2
        }
        
        # Execute
        sys_out, Rfirst, options_out = initReach_inputDependence(sys, Rinit, params, options)
        
        # Verify
        assert sys_out.taylorTerms == 4
        assert abs(sys_out.stepSize - 0.1) < 1e-10
        assert 'ti' in Rfirst
        assert 'tp' in Rfirst
        assert isinstance(Rfirst['ti'], Zonotope)
        assert isinstance(Rfirst['tp'], Zonotope)
        
        # TODO: Add MATLAB I/O pair comparisons
        # MATLAB: Rfirst.ti center, generators
        # MATLAB: Rfirst.tp center, generators
    
    def test_initReach_inputDependence_02_interval_matrix_A(self):
        """Test Case 2: System with interval matrix A"""
        # TODO: Add MATLAB I/O pairs after running debug script
        
        from cora_python.matrixSet.intervalMatrix import IntervalMatrix
        
        # Setup
        A_int = IntervalMatrix(np.array([[0, 1], [-1, -0.5]]), np.array([[0.1, 0], [0, 0.1]]))
        B = np.array([[0], [1]])
        c = np.array([[0], [0]])
        sys = LinearParamSys(A_int, B, c, 'constParam')
        
        # Initial set
        Rinit = Zonotope(np.array([[0], [0]]), np.array([[0.1, 0], [0, 0.1]]))
        
        # Parameters
        params = {
            'Uconst': Zonotope(np.array([[0], [0]]), np.array([[0.05, 0], [0, 0.05]])),
            'uTrans': np.array([[0.1], [0]])
        }
        
        # Options
        options = {
            'timeStep': 0.05,
            'taylorTerms': 3,
            'reductionTechnique': 'girard',
            'zonotopeOrder': 10,
            'compTimePoint': True,
            'intermediateTerms': 2
        }
        
        # Execute
        sys_out, Rfirst, options_out = initReach_inputDependence(sys, Rinit, params, options)
        
        # Verify
        assert sys_out.taylorTerms == 3
        assert abs(sys_out.stepSize - 0.05) < 1e-10
        assert 'ti' in Rfirst
        assert isinstance(Rfirst['ti'], Zonotope)
        
        # TODO: Add MATLAB I/O pair comparisons


if __name__ == "__main__":
    pytest.main([__file__])
