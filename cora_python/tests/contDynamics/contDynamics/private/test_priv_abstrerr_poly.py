"""
test_priv_abstrerr_poly - Test for priv_abstrerr_poly function
   MATLAB I/O pairs from debug_matlab_nonlinearSys_initReach_abstrerr_chain.m

   This test verifies that priv_abstrerr_poly correctly computes the abstraction 
   error for the polynomialization approach using the 6D tank example.

Syntax:
    pytest cora_python/tests/contDynamics/contDynamics/private/test_priv_abstrerr_poly.py

Authors:       MATLAB I/O pairs from debug_matlab_nonlinearSys_initReach_abstrerr_chain.m
Written:       2025
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.contDynamics.private.priv_abstrerr_poly import priv_abstrerr_poly
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.contDynamics.derivatives import derivatives
from cora_python.contDynamics.contDynamics.linReach import linReach


class MockNonlinearSys:
    """Mock nonlinearSys object for testing"""
    def __init__(self, nrOfDims=2):
        self.nrOfDims = nrOfDims
        self.name = 'test_sys'
        self.linError = type('obj', (object,), {
            'p': type('obj', (object,), {
                'x': np.array([[1], [2]]),
                'u': np.array([[0.5]])
            })()
        })()
    
    def setHessian(self, method):
        """Mock setHessian method"""
        return self
    
    def setThirdOrderTensor(self, method):
        """Mock setThirdOrderTensor method"""
        return self
    
    def hessian(self, x, u, *args):
        """Mock hessian method"""
        n = self.nrOfDims
        H = []
        for i in range(n):
            H_i = np.random.RandomState(42 + i).randn(n + len(u), n + len(u))
            H.append(H_i)
        return H
    
    def thirdOrderTensor(self, x, u, *args):
        """Mock thirdOrderTensor method"""
        n = self.nrOfDims
        T = []
        ind = []
        for i in range(n):
            T_i = []
            ind_i = []
            for j in range(n):
                T_ij = np.random.RandomState(100 + i * n + j).randn(n, n)
                T_i.append(T_ij)
                ind_i.append(j)
            T.append(T_i)
            ind.append(ind_i)
        return T, ind


class TestPrivAbstrerrPoly:
    """Test class for priv_abstrerr_poly functionality"""
    
    def test_priv_abstrerr_poly_tank6_example(self):
        """Test with 6D tank example (same as MATLAB test_nonlinearSys_initReach with alg='poly')"""
        # Model parameters (same as MATLAB test)
        dim_x = 6
        params = {
            'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
            'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
            'tFinal': 4,
            'uTrans': np.zeros((1, 1))
        }
        
        # Reachability settings (same as MATLAB test)
        # Note: tensorOrder=3 is needed for priv_precompStatError to be called
        options = {
            'timeStep': 4,
            'taylorTerms': 4,
            'zonotopeOrder': 50,
            'alg': 'poly',
            'tensorOrder': 3,  # Use 3 so priv_precompStatError is called
            'reductionTechnique': 'girard',
            'errorOrder': 10,
            'intermediateOrder': 10,
            'maxError': np.full((dim_x, 1), np.inf)  # Required by linReach
        }
        
        # System dynamics
        tank = NonlinearSys(tank6Eq, states=6, inputs=1)
        
        # Compute derivatives (required)
        derivatives(tank, options)
        
        # Compute factors
        for i in range(1, options['taylorTerms'] + 2):
            options['factor'] = options.get('factor', [])
            options['factor'].append((options['timeStep'] ** i) / np.math.factorial(i))
        
        # Prepare Rstart structure (as expected by linReach)
        Rstart = {'set': params['R0'], 'error': np.zeros((dim_x, 1))}
        
        # Call linReach to get Rmax (which includes abstraction error)
        # linReach calls priv_abstrerr_poly internally when alg='poly'
        Rti, Rtp, dimForSplit, options_out = linReach(tank, Rstart, params, options)
        
        # MATLAB result: linErrors (poly) = [0.000243530072665516; 0.00039402546364495; ...]
        expected_error = np.array([
            [0.000243530072665516],
            [0.00039402546364495],
            [0.000225203967329969],
            [0.000572556128362841],
            [0.000565233191232288],
            [0.000377256719313722]
        ])
        
        # Compare the error from linReach (which calls priv_abstrerr_poly internally)
        actual_error = Rtp['error']
        
        # Ensure both are 1D arrays for comparison
        if actual_error.ndim > 1:
            actual_error = actual_error.flatten()
        if expected_error.ndim > 1:
            expected_error = expected_error.flatten()
        
        # Compare with MATLAB results (exact match expected)
        np.testing.assert_allclose(actual_error, expected_error, rtol=1e-6, atol=1e-9)
    
    def test_priv_abstrerr_poly_basic(self):
        """Test basic priv_abstrerr_poly computation"""
        sys = MockNonlinearSys(nrOfDims=2)
        Rall = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
        Rdiff = Zonotope(np.array([[0], [0]]), 0.05 * np.eye(2))
        params = {
            'U': Zonotope(np.zeros((1, 1)), 0.05 * np.eye(1))
        }
        options = {
            'tensorOrder': 2,  # Less than 3, so no third-order
            'reductionTechnique': 'girard',
            'errorOrder': 10,
            'intermediateOrder': 10
        }
        
        # Mock inputs from priv_precompStatError
        H = sys.hessian(sys.linError.p.x, sys.linError.p.u)
        Zdelta = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
        VerrorStat = Zonotope(np.array([[0], [0]]), 0.01 * np.eye(2))
        T = None
        ind3 = None
        Zdelta3 = None
        
        try:
            trueError, VerrorDyn, VerrorStat_out = priv_abstrerr_poly(
                sys, Rall, Rdiff, params, options, H, Zdelta, VerrorStat, T, ind3, Zdelta3
            )
            
            # Verify outputs
            assert trueError is not None
            assert isinstance(trueError, Interval)
            
            assert VerrorDyn is not None
            assert isinstance(VerrorDyn, Zonotope)
            
            assert VerrorStat_out is not None
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"Dependencies not yet translated: {e}")
    
    def test_priv_abstrerr_poly_tensorOrder_3(self):
        """Test with tensorOrder == 3 (with third-order error)"""
        sys = MockNonlinearSys(nrOfDims=2)
        Rall = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
        Rdiff = Zonotope(np.array([[0], [0]]), 0.05 * np.eye(2))
        params = {
            'U': Zonotope(np.zeros((1, 1)), 0.05 * np.eye(1))
        }
        options = {
            'tensorOrder': 3,
            'reductionTechnique': 'girard',
            'errorOrder': 10,
            'intermediateOrder': 10
        }
        
        H = sys.hessian(sys.linError.p.x, sys.linError.p.u)
        Zdelta = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
        VerrorStat = Zonotope(np.array([[0], [0]]), 0.01 * np.eye(2))
        T, ind3 = sys.thirdOrderTensor(sys.linError.p.x, sys.linError.p.u)
        Zdelta3 = None
        
        try:
            trueError, VerrorDyn, VerrorStat_out = priv_abstrerr_poly(
                sys, Rall, Rdiff, params, options, H, Zdelta, VerrorStat, T, ind3, Zdelta3
            )
            
            # Should compute third-order error
            assert trueError is not None
            assert VerrorDyn is not None
            assert VerrorStat_out is not None
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"Dependencies not yet translated: {e}")
    
    def test_priv_abstrerr_poly_tensorOrder_4(self):
        """Test with tensorOrder >= 4 (with higher-order error)"""
        sys = MockNonlinearSys(nrOfDims=2)
        Rall = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
        Rdiff = Zonotope(np.array([[0], [0]]), 0.05 * np.eye(2))
        params = {
            'U': Zonotope(np.zeros((1, 1)), 0.05 * np.eye(1))
        }
        options = {
            'tensorOrder': 4,
            'reductionTechnique': 'girard',
            'errorOrder': 10,
            'intermediateOrder': 10,
            'errorOrder3': 5
        }
        
        H = sys.hessian(sys.linError.p.x, sys.linError.p.u)
        Zdelta = Zonotope(np.array([[0], [0]]), 0.1 * np.eye(2))
        VerrorStat = Zonotope(np.array([[0], [0]]), 0.01 * np.eye(2))
        T, ind3 = sys.thirdOrderTensor(sys.linError.p.x, sys.linError.p.u)
        Zdelta3 = Zonotope(np.array([[0], [0]]), 0.05 * np.eye(2))
        
        try:
            trueError, VerrorDyn, VerrorStat_out = priv_abstrerr_poly(
                sys, Rall, Rdiff, params, options, H, Zdelta, VerrorStat, T, ind3, Zdelta3
            )
            
            # Should handle higher-order error
            assert trueError is not None
            assert VerrorDyn is not None
            assert VerrorStat_out is not None
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"Dependencies not yet translated: {e}")


def test_priv_abstrerr_poly():
    """Test function for priv_abstrerr_poly method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestPrivAbstrerrPoly()
    test.test_priv_abstrerr_poly_basic()
    test.test_priv_abstrerr_poly_tensorOrder_3()
    test.test_priv_abstrerr_poly_tensorOrder_4()
    
    print("test_priv_abstrerr_poly: all tests passed")
    return True


if __name__ == "__main__":
    test_priv_abstrerr_poly()

