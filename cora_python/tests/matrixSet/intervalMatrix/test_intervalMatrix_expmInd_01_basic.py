"""
Test file for intervalMatrix expmInd - translated from MATLAB

This test verifies the expmInd function for intervalMatrix.

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.matrixSet.intervalMatrix import IntervalMatrix
from cora_python.matrixSet.intervalMatrix.expmInd import expmInd

# MATLAB I/O pairs from debug_matlab_expmInd_intervalMatrix.m
tol = 1e-10


class TestIntervalMatrixExpmInd:
    """Test cases for intervalMatrix expmInd"""
    
    def test_expmInd_01_simple_2x2(self):
        """Test Case 1: Simple 2x2 interval matrix"""
        # Setup
        C = np.array([[0, 1], [-1, -0.5]])
        D = np.array([[0.1, 0], [0, 0.1]])
        intMat = IntervalMatrix(C, D)
        
        maxOrder = 4
        
        # Execute
        eI = expmInd(intMat, maxOrder)
        
        # Verify types
        assert isinstance(eI, IntervalMatrix)
        
        # MATLAB I/O pairs from debug_matlab_expmInd_intervalMatrix.m
        # eI center (from MATLAB):
        eI_center_expected = np.array([[0.6138, 0.6650], [-0.6633, 0.2878]])
        np.testing.assert_allclose(eI.int.center(), eI_center_expected, rtol=tol, atol=tol)
        
        # eI radius (from MATLAB):
        eI_rad_expected = np.array([[0.1848, 0.1722], [0.1722, 0.2545]])
        np.testing.assert_allclose(eI.int.rad(), eI_rad_expected, rtol=tol, atol=tol)
    
    def test_expmInd_02_with_initialOrder_and_initialPower(self):
        """Test Case 2: With initialOrder and initialPower"""
        # Setup
        C = np.array([[0, 1], [-1, -0.5]])
        D = np.array([[0.1, 0], [0, 0.1]])
        intMat = IntervalMatrix(C, D)
        
        maxOrder2 = 3
        initialOrder = 2
        initialPower = intMat @ intMat  # intMat^2
        
        # Execute
        eI2, iPow2, E2 = expmInd(intMat, maxOrder2, initialOrder, initialPower)
        
        # Verify types
        assert isinstance(eI2, IntervalMatrix)
        assert isinstance(E2, IntervalMatrix)
        assert isinstance(iPow2, list)
        assert len(iPow2) == 3
        
        # MATLAB I/O pairs from debug_matlab_expmInd_intervalMatrix.m
        # eI2 center (from MATLAB):
        eI2_center_expected = np.array([[-0.4167, -0.3717], [0.3733, -0.2233]])
        np.testing.assert_allclose(eI2.int.center(), eI2_center_expected, rtol=tol, atol=tol)
        
        # eI2 radius (from MATLAB):
        eI2_rad_expected = np.array([[0.1378, 0.2224], [0.2241, 0.2345]])
        np.testing.assert_allclose(eI2.int.rad(), eI2_rad_expected, rtol=tol, atol=tol)
        
        # E2 radius (from MATLAB):
        E2_rad_expected = np.array([[0.0826, 0.0957], [0.0957, 0.1305]])
        np.testing.assert_allclose(E2.int.rad(), E2_rad_expected, rtol=tol, atol=tol)
    
    def test_expmInd_03_3x3(self):
        """Test Case 3: 3x3 interval matrix"""
        # Setup
        C3 = np.array([[0, 1, 0], [-1, -0.5, 0], [0, 0, -1]])
        D3 = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        intMat3 = IntervalMatrix(C3, D3)
        
        maxOrder3 = 3
        
        # Execute
        eI3 = expmInd(intMat3, maxOrder3)
        
        # Verify types
        assert isinstance(eI3, IntervalMatrix)
        
        # MATLAB I/O pairs from debug_matlab_expmInd_intervalMatrix.m
        # eI3 center (from MATLAB):
        eI3_center_expected = np.array([[0.5833, 0.6283, 0], [-0.6267, 0.2767, 0], [0, 0, 0.3333]])
        np.testing.assert_allclose(eI3.int.center(), eI3_center_expected, rtol=tol, atol=tol)
        
        # eI3 radius (from MATLAB):
        eI3_rad_expected = np.array([[0.2378, 0.2224, 0], [0.2241, 0.3345, 0], [0, 0, 0.3275]])
        np.testing.assert_allclose(eI3.int.rad(), eI3_rad_expected, rtol=tol, atol=tol)


if __name__ == "__main__":
    pytest.main([__file__])
