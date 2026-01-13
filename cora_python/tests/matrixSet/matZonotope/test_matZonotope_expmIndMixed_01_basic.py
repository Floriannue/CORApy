"""
Test file for expmIndMixed - translated from MATLAB

This test verifies the expmIndMixed function for matZonotope.

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.matrixSet.matZonotope import matZonotope
from cora_python.matrixSet.matZonotope.expmIndMixed import expmIndMixed
from cora_python.matrixSet.intervalMatrix import IntervalMatrix

# MATLAB I/O pairs from debug_matlab_expmIndMixed.m
tol = 1e-10


class TestExpmIndMixed:
    """Test cases for expmIndMixed"""
    
    def test_expmIndMixed_01_simple_2x2(self):
        """Test Case 1: Simple 2x2 matrix zonotope"""
        # Setup
        C = np.array([[0, 1], [-1, -0.5]])
        G = np.zeros((2, 2, 2))
        G[:, :, 0] = np.array([[0.1, 0], [0, 0.1]])
        G[:, :, 1] = np.array([[0, 0.05], [0.05, 0]])
        matZ = matZonotope(C, G)
        
        intermediateOrder = 2
        maxOrder = 4
        
        # Execute
        eZ, eI, zPow, iPow, E = expmIndMixed(matZ, intermediateOrder, maxOrder)
        
        # Verify types
        assert isinstance(eZ, matZonotope)
        assert isinstance(eI, IntervalMatrix)
        assert isinstance(E, IntervalMatrix)
        assert isinstance(zPow, list)
        assert isinstance(iPow, list)
        assert len(zPow) == 2
        assert len(iPow) == 4
        
        # MATLAB I/O pairs from debug_matlab_expmIndMixed.m
        # eI center (from MATLAB):
        eI_center_expected = np.array([[0.1177, -0.0875], [0.0833, 0.1618]])
        np.testing.assert_allclose(eI.int.center(), eI_center_expected, rtol=tol, atol=tol)
        
        # eI radius (from MATLAB):
        eI_rad_expected = np.array([[0.1045, 0.1125], [0.1118, 0.1409]])
        np.testing.assert_allclose(eI.int.rad(), eI_rad_expected, rtol=tol, atol=tol)
        
        # E should be small (from MATLAB: ~0.02-0.04)
        E_rad = E.int.rad()
        assert np.all(E_rad < 0.05)
    
    def test_expmIndMixed_02_3x3(self):
        """Test Case 2: 3x3 matrix zonotope"""
        # Setup
        C2 = np.array([[0, 1, 0], [-1, -0.5, 0], [0, 0, -1]])
        G2 = np.zeros((3, 3, 1))
        G2[:, :, 0] = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        matZ2 = matZonotope(C2, G2)
        
        intermediateOrder2 = 2
        maxOrder2 = 3
        
        # Execute
        eZ2, eI2, zPow2, iPow2, E2 = expmIndMixed(matZ2, intermediateOrder2, maxOrder2)
        
        # Verify types
        assert isinstance(eZ2, matZonotope)
        assert isinstance(eI2, IntervalMatrix)
        assert isinstance(E2, IntervalMatrix)
        
        # MATLAB I/O pairs from debug_matlab_expmIndMixed.m
        # eZ2 center (from MATLAB):
        eZ2_center_expected = np.array([[0.5000, 0.7500, 0], [-0.7500, 0.1250, 0], [0, 0, 0.5000]])
        np.testing.assert_allclose(eZ2.C, eZ2_center_expected, rtol=tol, atol=tol)
        
        # eI2 center (from MATLAB):
        eI2_center_expected = np.array([[0.0833, -0.1250, 0], [0.1217, 0.1477, 0], [0, 0, -0.1702]])
        np.testing.assert_allclose(eI2.int.center(), eI2_center_expected, rtol=tol, atol=tol)


if __name__ == "__main__":
    pytest.main([__file__])
