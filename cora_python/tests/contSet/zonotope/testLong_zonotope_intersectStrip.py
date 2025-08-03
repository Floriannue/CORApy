"""
testLong_zonotope_intersectStrip - unit test function of intersectStrip

Tests the intersectStrip function for zonotopes with various methods.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       08-September-2020 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope


def testLong_zonotope_intersectStrip():
    """Test long version of intersectStrip functionality"""
    # Assume true
    res = True
    
    # Check whether the over-approximation of intersectStrip encloses the
    # exact result
    
    # Test methods handling multiple strips at a time
    # Specify strips
    C = np.array([[1, 0], [0, 1], [1, 1]])
    phi = np.array([[5], [3], [3]])
    y = np.array([[-2], [2], [2]])
    
    # Polytope of intersected strips
    P = Polytope(
        np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]]),
        np.array([[3], [7], [5], [1], [5], [1]])
    )
    
    # Specify zonotope
    Z = Zonotope(np.array([[1], [1]]), np.array([[2, 2, 2, 6, 2, 8], [2, 2, 0, 5, 0, 6]]))
    
    # Obtain over-approximative zonotope after intersection
    Z_over = []
    Z_over.append(Z.intersectStrip(C, phi, y, 'normGen'))
    Z_over.append(Z.intersectStrip(C, phi, y, 'svd'))
    Z_over.append(Z.intersectStrip(C, phi, y, 'radius'))
    # Flag 'alamo-volume' is disabled for non-scalar phi:
    # Z_over.append(Z.intersectStrip(C, phi, y, 'alamo-volume'))
    
    # Obtain exact solution
    P_exact = Polytope(Z) & P
    
    # Testing methods for single strips
    # Specify strip
    C_single = np.array([[1, 0]])
    phi_single = np.array([[5]])
    y_single = np.array([[-2]])
    
    # Polytope of intersected strips
    P_single = Polytope(np.array([[1, 0], [-1, 0]]), np.array([[3], [7]]))
    
    # Obtain over-approximative zonotope after intersection
    Z_over_single = []
    Z_over_single.append(Z.intersectStrip(C_single, phi_single, y_single, 'normGen'))
    Z_over_single.append(Z.intersectStrip(C_single, phi_single, y_single, 'svd'))
    Z_over_single.append(Z.intersectStrip(C_single, phi_single, y_single, 'radius'))
    Z_over_single.append(Z.intersectStrip(C_single, phi_single, y_single, 'alamo-volume'))
    Z_over_single.append(Z.intersectStrip(C_single, phi_single, y_single, 'alamo-FRad'))
    Z_over_single.append(Z.intersectStrip(C_single, phi_single, y_single, 'bravo'))
    
    # Obtain exact solution
    P_exact_single = Polytope(Z) & P_single
    
    # Exact solution enclosed in over-approximation?
    # Specify tolerance
    tol = 1e-6
    Z_tol = Zonotope(np.zeros((2, 1)), tol * np.eye(2))
    
    # Case for multiple strips
    for i in range(len(Z_over)):
        # Check if polytope of Z_over[i] + Z_tol contains P_exact
        Z_enlarged = Z_over[i] + Z_tol
        P_enlarged = Polytope(Z_enlarged)
        assert P_enlarged.contains_(P_exact)
    
    # Case for single strip
    for i in range(len(Z_over_single)):
        # Check if polytope of Z_over_single[i] + Z_tol contains P_exact_single
        Z_enlarged = Z_over_single[i] + Z_tol
        P_enlarged = Polytope(Z_enlarged)
        assert P_enlarged.contains_(P_exact_single)
    
    # Optional: Create plots for visualization
    # figure; hold on 
    # plot(Z,[1 2],'r-+');
    # plot(P,[1 2],'r-*');
    # plot(Z_over[0],[1 2],'b-+');
    # plot(Z & P,[1 2],'g');
    # plot(P_exact,[1 2],'b-*');
    # legend('zonotope','strips','zonoStrips','zono&poly','exact');
    
    # figure; hold on 
    # plot(Z,[1 2],'r-+');
    # plot(P_single,[1 2],'r-*');
    # plot(Z_over_single[0],[1 2],'b-+');
    # plot(Z_over_single[4],[1 2],'g-');
    # plot(P_exact_single,[1 2],'b-*');
    # legend('zonotope','strips','zonoStrips','bravoMethod','exact');
    
    assert res


if __name__ == "__main__":
    testLong_zonotope_intersectStrip() 