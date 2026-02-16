"""
testLong_zonotope_minkDiff_2 - unit test function of minkDiff for
   approximating the Minkowski difference of two zonotopes or a zonotope
   with a vector according to [1].

This module contains long unit tests for the zonotope minkDiff method, which computes
the Minkowski difference of two zonotopes.

References:
   [1] M. Althoff, "On Computing the Minkowski Difference of Zonotopes"

Authors: Matthias Althoff
Written: 06-May-2021
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def testLong_zonotope_minkDiff_2():
    """Unit test function of minkDiff - mirrors MATLAB testLong_zonotope_minkDiff_2.m"""
    
    # Create zonotopes -- fixed cases in 2D (Minkowski difference is exact in 2D)
    Z_m = Zonotope(np.array([[1, 1, 0, 1], [1, 0, 1, 1]]).T)
    Z_m_degenerate = Zonotope(np.array([[1, 1], [1, 0]]).T)
    Z_s = [
        Zonotope(np.array([[0, 0.5, 0], [0, -0.2, 0.2]]).T),  # see Fig. 2a in [1]
        Zonotope(np.array([[0, 0.5, 0], [0, -0.5, 0.5]]).T),  # see Fig. 2b in [1]
        Zonotope(np.array([[0, 2, 0], [0, -0.5, 0.5]]).T),    # see Fig. 2c in [1]
        Zonotope(np.array([[0, 0.5, 0], [0, 0, 0.5]]).T),
        Zonotope(np.array([[0, 1, 0], [0, 0, 0.5]]).T),
        Zonotope(np.array([[0, 2, 0], [0, 0, 0.5]]).T)
    ]
    Z_s_degenerate = Zonotope(np.array([[1, 0.5], [1, 0]]).T)
    
    # Convert Z_m to polytope
    P_m = Polytope(Z_m)
    P_m_degenerate = Polytope(Z_m_degenerate)
    
    # Define small box
    smallBox = Zonotope(np.array([[0], [0]]), 1e-8 * np.eye(2))
    verySmallBox = Zonotope(np.array([[0], [0]]), 1e-12 * np.eye(2))
    
    # Loop through all subtrahends and check for exact result in the 2D case
    for iSet, Z_subtrahend in enumerate(Z_s):
        
        # Result from polytopes
        # IMPORTANT: Minkowski difference of MPT toolbox seems to be incorrect
        # for minkDiff(Z_s{7},Z_s{9})
        P_res = P_m.minkDiff(Polytope(Z_subtrahend))
        
        # Set considered types
        typeSet = ['inner', 'outer']
        
        # Loop over all types
        for type_method in typeSet:
            
            # Compute result
            Z_res = Z_m.minkDiff(Z_subtrahend, type_method)
            
            # Check whether Minkowski difference returns the empty set
            if Z_res.representsa('emptySet'):
                # Check if polytope solution is empty as well
                assert P_res.representsa('emptySet')
            else:
                # Enclosure check (Definition of Minkowski difference)
                assert (Z_m + smallBox).contains(Z_res + Z_subtrahend)
                
                # Enclosure check (comparison with polytope solution)
                assert (P_res + smallBox).contains(Polytope(Z_res))
                
                # Enclosure check (comparison with polytope solution; other direction)
                assert (Polytope(Z_res) + smallBox).contains(P_res + verySmallBox)
    
    # Minuend is a degenerate zonotope and the subtrahend is not
    # Result from polytopes
    P_res = P_m_degenerate.minkDiff(Polytope(Z_s[0]))
    
    # Set considered types
    typeSet = ['inner', 'outer', 'outer:coarse', 'inner:conZonotope', 
               'inner:RaghuramanKoeln']
    
    # Loop over all types
    for type_method in typeSet:
        
        # Compute result
        Z_res = Z_m_degenerate.minkDiff(Z_s[0], type_method)
        
        # The result should be empty
        if Z_res.representsa('emptySet'):
            # Check if polytope solution is empty as well
            assert P_res.representsa('emptySet')
    
    # Minuend and subtrahend are degenerate in 2D (result should be exact)
    # Result from polytopes
    P_res = P_m_degenerate.minkDiff(Polytope(Z_s_degenerate))
    
    # Set considered types
    typeSet = ['inner', 'outer', 'outer:coarse']
    
    # Loop over all types
    for type_method in typeSet:
        
        # Compute result
        Z_res = Z_m_degenerate.minkDiff(Z_s_degenerate, type_method)
        
        # Enclosure check (Definition of Minkowski difference)
        assert (Z_m_degenerate + smallBox).contains(Z_res + Z_s_degenerate)
        
        # Enclosure check (comparison with polytope solution)
        assert (P_res + smallBox).contains(Polytope(Z_res))
        
        # Enclosure check (comparison with polytope solution; other direction)
        assert (Polytope(Z_res) + smallBox).contains(P_res)
    
    # Create zonotopes -- fixed case in 3D (no halfspace is redundant; check under-approximation)
    # Define small box
    smallBox = Zonotope(np.array([[0], [0], [0]]), 1e-6 * np.eye(3))
    # Create minuend
    Z_m = Zonotope(np.zeros((3, 1)), np.hstack([np.ones((3, 1)), np.eye(3)]))
    # Create subtrahend
    Z_s = 1/4 * Zonotope(np.zeros((3, 1)), np.array([[-1], [1], [1]]))
    
    # Set considered types
    typeSet = ['inner', 'inner:conZonotope', 'inner:RaghuramanKoeln']
    
    # Loop over all types
    for type_method in typeSet:
        
        # Compute result
        Z_res = Z_m.minkDiff(Z_s, type_method)
        
        # Enclosure check (Definition of Minkowski difference)
        assert (Z_m + smallBox).contains(Z_res + Z_s)
    
    # Create zonotopes -- random cases in 3D (check under-approximation)
    for iSet in range(10):
        # Create minuend
        Z_m = Zonotope.generateRandom(dimension=3, nr_generators=5)
        
        # Create subtrahend
        Z_s = Zonotope.generateRandom(dimension=3, nr_generators=5).enlarge(0.2)
        
        # Set considered types
        typeSet = ['inner', 'inner:conZonotope', 'inner:RaghuramanKoeln']
        
        # Loop over all types
        for type_method in typeSet:
            
            # Compute result
            Z_res = Z_m.minkDiff(Z_s, type_method)
            
            # Enclosure check (Definition of Minkowski difference)
            assert (Z_m + smallBox).contains(Z_res + Z_s)


if __name__ == '__main__':
    testLong_zonotope_minkDiff_2()
    print("testLong_zonotope_minkDiff_2 passed!") 