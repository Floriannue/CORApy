"""
Test file for expmOneParam - translated from MATLAB

This test verifies the expmOneParam function for matZonotope.

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.matrixSet.matZonotope import matZonotope
from cora_python.matrixSet.matZonotope.expmOneParam import expmOneParam
from cora_python.contSet.zonotope import Zonotope
from cora_python.matrixSet.intervalMatrix import IntervalMatrix

# MATLAB I/O pairs from debug_matlab_expmOneParam.m
# Note: MATLAB debug output shows 4 decimal places, but full precision may have small differences
tol = 1e-3  # Numerical precision tolerance for RconstInput (small differences expected)


class TestExpmOneParam:
    """Test cases for expmOneParam"""
    
    def test_expmOneParam_01_simple_2x2(self):
        """Test Case 1: Simple 2x2 matrix zonotope with one generator"""
        # Setup
        C = np.array([[0, 1], [-1, -0.5]])
        G = np.zeros((2, 2, 1))
        G[:, :, 0] = np.array([[0.1, 0], [0, 0.1]])
        matZ = matZonotope(C, G)
        
        r = 0.1
        maxOrder = 4
        params = {
            'Uconst': Zonotope(np.array([[0], [0]]), np.array([[0.05, 0], [0, 0.05]])),
            'uTrans': np.array([[0.1], [0]])
        }
        
        # Execute
        eZ, eI, zPow, iPow, E, RconstInput = expmOneParam(matZ, r, maxOrder, params)
        
        # Verify types
        assert isinstance(eZ, matZonotope)
        assert isinstance(eI, IntervalMatrix)
        assert isinstance(E, IntervalMatrix)
        assert isinstance(RconstInput, Zonotope)
        assert isinstance(zPow, list)
        assert isinstance(iPow, list)
        assert len(zPow) == 4
        assert len(iPow) == 0
        
        # MATLAB I/O pairs from debug_matlab_expmOneParam.m
        # eZ center (from MATLAB full precision, verified by running MATLAB):
        # MATLAB: 0.995086750000000 0.097880458333333
        #         -0.097880458333333 0.946146520833333
        eZ_center_expected = np.array([[0.995086750000000, 0.097880458333333], 
                                       [-0.097880458333333, 0.946146520833333]])
        np.testing.assert_allclose(eZ.C, eZ_center_expected, rtol=tol, atol=tol)
        
        # RconstInput center (from MATLAB full precision, verified by running MATLAB):
        # MATLAB: 0.010008562500000; -0.000491329166667
        RconstInput_center_expected = np.array([[0.010008562500000], [-0.000491329166667]])
        np.testing.assert_allclose(RconstInput.center(), RconstInput_center_expected, rtol=tol, atol=tol)
        
        # E should be very small (from MATLAB: ~1e-6)
        E_center = E.int.center()
        E_rad = E.int.rad()
        assert np.all(np.abs(E_center) < 1e-5)
        assert np.all(E_rad < 1e-5)
    
    def test_expmOneParam_02_3x3(self):
        """Test Case 2: 3x3 matrix zonotope"""
        # Setup
        C2 = np.array([[0, 1, 0], [-1, -0.5, 0], [0, 0, -1]])
        G2 = np.zeros((3, 3, 1))
        G2[:, :, 0] = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        matZ2 = matZonotope(C2, G2)
        
        r2 = 0.05
        maxOrder2 = 3
        params2 = {
            'Uconst': Zonotope(np.array([[0], [0], [0]]), np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])),
            'uTrans': np.array([[0.1], [0], [0]])
        }
        
        # Execute
        eZ2, eI2, zPow2, iPow2, E2, RconstInput2 = expmOneParam(matZ2, r2, maxOrder2, params2)
        
        # Verify types
        assert isinstance(eZ2, matZonotope)
        assert isinstance(eI2, IntervalMatrix)
        assert isinstance(E2, IntervalMatrix)
        assert isinstance(RconstInput2, Zonotope)
        assert len(zPow2) == 3
        
        # MATLAB I/O pairs from debug_matlab_expmOneParam.m
        # eZ2 center (from MATLAB):
        eZ2_center_expected = np.array([[0.9988, 0.0495, 0], [-0.0495, 0.9740, 0], [0, 0, 0.9511]])
        np.testing.assert_allclose(eZ2.C, eZ2_center_expected, rtol=tol, atol=tol)
        
        # eZ2 generators shape (from MATLAB): [3 3 3]
        assert eZ2.G.shape == (3, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__])
