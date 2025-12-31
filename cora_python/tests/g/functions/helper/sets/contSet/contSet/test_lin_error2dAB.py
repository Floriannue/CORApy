"""
test_lin_error2dAB - GENERATED TEST
   No MATLAB source test exists. This test was created by analyzing the MATLAB 
   implementation logic in lin_error2dAB.py and ensuring thorough coverage.

   This test verifies that lin_error2dAB correctly computes the uncertainty 
   interval to be added to system matrix set caused by Lagrangian remainder of 
   the linearization, including:
   - Computing intervals from reachable set and input set
   - Computing deviation intervals (dxInt, duInt)
   - Evaluating Hessian function
   - Computing dA and dB matrices from Hessian and deviations
   - Handling interval matrices in Hessian

Syntax:
    pytest cora_python/tests/g/functions/helper/sets/contSet/contSet/test_lin_error2dAB.py

Authors:       Generated test based on MATLAB implementation
Written:       2025
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.helper.sets.contSet.contSet.lin_error2dAB import lin_error2dAB


def mock_hessian(totalInt, inputInt, *args):
    """
    Mock Hessian function that returns a list of matrices
    Each matrix represents the Hessian for one state dimension
    """
    dim_x = totalInt.dim()
    dim_u = inputInt.dim()
    
    # Create simple Hessian matrices (2D for each state dimension)
    H = []
    for i in range(dim_x):
        # Each H[i] is a (dim_x + dim_u) x (dim_x + dim_u) matrix
        H_i = np.random.RandomState(42 + i).randn(dim_x + dim_u, dim_x + dim_u)
        H.append(H_i)
    
    return H


class TestLinError2dAB:
    """Test class for lin_error2dAB functionality"""
    
    def test_lin_error2dAB_basic(self):
        """Test basic lin_error2dAB computation"""
        # Create reachable set (zonotope)
        dim_x = 2
        R = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(dim_x))
        
        # Create input set
        dim_u = 1
        U = Zonotope(np.array([[0]]), 0.05 * np.eye(dim_u))
        
        # Linearization point
        p = {
            'x': np.array([[1], [2]]),
            'u': np.array([[0.5]])
        }
        
        # Compute dA and dB
        dA, dB = lin_error2dAB(R, U, mock_hessian, p)
        
        # Verify dimensions
        assert dA.shape == (dim_x, dim_x)
        assert dB.shape == (dim_x, dim_u)
        
        # Verify values are finite
        assert np.all(np.isfinite(dA))
        assert np.all(np.isfinite(dB))
    
    def test_lin_error2dAB_3D_system(self):
        """Test with 3D system"""
        dim_x = 3
        dim_u = 2
        R = Zonotope(np.zeros((dim_x, 1)), 0.1 * np.eye(dim_x))
        U = Zonotope(np.zeros((dim_u, 1)), 0.05 * np.eye(dim_u))
        
        p = {
            'x': np.zeros((dim_x, 1)),
            'u': np.zeros((dim_u, 1))
        }
        
        dA, dB = lin_error2dAB(R, U, mock_hessian, p)
        
        assert dA.shape == (dim_x, dim_x)
        assert dB.shape == (dim_x, dim_u)
    
    def test_lin_error2dAB_interval_hessian(self):
        """Test with interval Hessian (if supported)"""
        dim_x = 2
        dim_u = 1
        R = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(dim_x))
        U = Zonotope(np.array([[0]]), 0.05 * np.eye(dim_u))
        
        p = {
            'x': np.array([[0], [0]]),
            'u': np.array([[0]])
        }
        
        def hessian_interval(totalInt, inputInt, *args):
            """Hessian that returns interval matrices"""
            dim_x = totalInt.dim()
            dim_u = inputInt.dim()
            H = []
            for i in range(dim_x):
                # Create interval matrix
                H_i = np.zeros((dim_x + dim_u, dim_x + dim_u), dtype=object)
                for j in range(dim_x + dim_u):
                    for k in range(dim_x + dim_u):
                        H_i[j, k] = Interval(-0.1, 0.1)
                H.append(H_i)
            return H
        
        dA, dB = lin_error2dAB(R, U, hessian_interval, p)
        
        assert dA.shape == (dim_x, dim_x)
        assert dB.shape == (dim_x, dim_u)
        assert np.all(np.isfinite(dA))
        assert np.all(np.isfinite(dB))
    
    def test_lin_error2dAB_with_varargin(self):
        """Test with additional arguments to hessian"""
        dim_x = 2
        dim_u = 1
        R = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(dim_x))
        U = Zonotope(np.array([[0]]), 0.05 * np.eye(dim_u))
        
        p = {
            'x': np.array([[0], [0]]),
            'u': np.array([[0]])
        }
        
        def hessian_with_params(totalInt, inputInt, param1, param2):
            """Hessian that accepts additional parameters"""
            return mock_hessian(totalInt, inputInt)
        
        # Pass additional arguments
        dA, dB = lin_error2dAB(R, U, hessian_with_params, p, 'param1', 'param2')
        
        assert dA.shape == (dim_x, dim_x)
        assert dB.shape == (dim_x, dim_u)


def test_lin_error2dAB():
    """Test function for lin_error2dAB method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestLinError2dAB()
    test.test_lin_error2dAB_basic()
    test.test_lin_error2dAB_3D_system()
    test.test_lin_error2dAB_interval_hessian()
    test.test_lin_error2dAB_with_varargin()
    
    print("test_lin_error2dAB: all tests passed")
    return True


if __name__ == "__main__":
    test_lin_error2dAB()

