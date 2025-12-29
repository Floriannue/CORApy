"""
testLong_contract - unit test function for contractors

Syntax:
    res = testLong_contract()

Inputs:
    -

Outputs:
    res - true/false 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contract, contractPoly

Authors:       Niklas Kochdumper
Written:       18-December-2020
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractPoly import contractPoly
from cora_python.contSet.interval.contains_ import contains_


def testLong_contract():
    """
    Unit test function for contractors
    """
    tol = 1e-10
    
    # Test 1: Different Contractors -------------------------------------------
    
    # Test if all contractors that are implemented provide a valid result for
    # contraction
    
    # Contraction problem
    def f(x):
        return x[0]**2 + x[1]**2 - 4
    
    dom = Interval([1, 1], [3, 3])
    
    # Optimal result
    Iopt = Interval([1, 1], [np.sqrt(3), np.sqrt(3)])
    
    # Contractors
    cont = ['forwardBackward', 'polynomial', 'linearize', 'interval', 'all']
    iter_val = 2
    splits = 2
    
    # Loop over all contractors
    # MATLAB: I1 = contract(f,dom,cont{i},iter,splits);
    for i, alg in enumerate(cont):
        # Contract the domain
        I1 = contract(f, dom, alg, iter_val, splits)
        
        # Check if result contains optimal
        assert I1 is not None, f"Contractor {alg} returned None"
        assert contains_(I1, Iopt, 'exact', tol), \
            f"Contractor {alg} failed: result does not contain optimal interval"
    
    # Test 2: "contract" vs. "contractPoly" -----------------------------------
    
    # Test if "contract" and "contractPoly" give the same result if they are
    # called for the same contraction problem
    
    # Contraction problem
    def f2(x):
        return x[0]**2 + x[1]**2 - 4
    
    dom2 = Interval([1, 1], [3, 3])
    
    # Equivalent formulation with a polynomial function
    c2 = -4
    G2 = np.array([[1, 1]])
    GI2 = np.array([])
    E2 = 2 * np.eye(2)
    
    # Contractors
    cont2 = ['forwardBackward', 'polynomial', 'linearize', 'interval', 'all']
    iter_val2 = 2
    splits2 = 2
    
    # Loop over all contractors
    # MATLAB: I1 = contract(f,dom,cont{i},iter,splits);
    # MATLAB: I2 = contractPoly(c,G,GI,E,dom,cont{i},iter,splits);
    for i, alg in enumerate(cont2):
        # Contract with "contract"
        I1 = contract(f2, dom2, alg, iter_val2, splits2)
        
        # Contract with "contractPoly"
        I2 = contractPoly(c2, G2, GI2, E2, dom2, alg, iter_val2, splits2)
        
        # Check the result for correctness
        # MATLAB: assertLoop(isequal(I1,I2,tol),i);
        assert I1 is not None and I2 is not None, \
            f"Contractor {alg}: one result is None"
        assert np.allclose(I1.inf, I2.inf, atol=tol), \
            f"Contractor {alg}: infimum mismatch"
        assert np.allclose(I1.sup, I2.sup, atol=tol), \
            f"Contractor {alg}: supremum mismatch"
    
    # Test 3: Multiple Constraints --------------------------------------------
    
    # Contraction problem
    def f3(x):
        return np.array([x[0]**2 - 4*x[1], x[1]**2 - 2*x[0] + 4*x[1]])
    
    dom3 = Interval([-0.1, -0.1], [0.1, 0.1])
    
    # Equivalent formulation with a polynomial function
    c3 = np.array([[0], [0]])
    G3 = np.array([[1, 0, 0, -4], [0, 1, -2, 4]])
    E3 = np.array([[2, 0, 1, 0], [0, 2, 0, 1]])
    GI3 = np.array([])
    
    # Contractors
    cont3 = ['forwardBackward', 'polynomial', 'linearize', 'interval', 'all']
    iter_val3 = 2
    splits3 = 2
    
    # Loop over all contractors
    # MATLAB: I1 = contract(f,dom,cont{i},iter,splits);
    # MATLAB: I2 = contractPoly(c,G,GI,E,dom,cont{i},iter,splits);
    for i, alg in enumerate(cont3):
        # Contract with "contract"
        I1 = contract(f3, dom3, alg, iter_val3, splits3)
        
        # Contract with "contractPoly"
        I2 = contractPoly(c3, G3, GI3, E3, dom3, alg, iter_val3, splits3)
        
        # Check the result for correctness
        # MATLAB: assertLoop(isequal(I1,I2,tol),i)
        assert I1 is not None and I2 is not None, \
            f"Contractor {alg}: one result is None"
        assert np.allclose(I1.inf, I2.inf, atol=tol), \
            f"Contractor {alg}: infimum mismatch"
        assert np.allclose(I1.sup, I2.sup, atol=tol), \
            f"Contractor {alg}: supremum mismatch"
    
    return True


if __name__ == '__main__':
    testLong_contract()
    print("All tests passed!")

