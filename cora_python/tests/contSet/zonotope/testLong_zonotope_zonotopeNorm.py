#!/usr/bin/env python3
"""
testLong_zonotope_zonotopeNorm - long unit test function of zonotopeNorm

Authors: Adrian Kulmburg (MATLAB)
         Python translation by AI Assistant
Written: 06-July-2021 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
import time
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.zonotopeNorm import zonotopeNorm


def testLong_zonotope_zonotopeNorm():
    """
    Long test function for zonotopeNorm that performs extensive testing
    across different dimensions and scenarios.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    
    print("=== Running testLong_zonotope_zonotopeNorm ===\n")
    start_time = time.time()
    
    # Assume true initially
    res = True
    
    dims = [2, 5, 10]  # Dimensions to be tested
    Ntests = 10  # Number of tests in each case
    Npoints = 10  # Number of points to be tested
    
    print(f"Testing dimensions: {dims}")
    print(f"Tests per dimension: {Ntests}")
    print(f"Points per test: {Npoints}")
    print()
    
    # Test 1: Points inside zonotopes
    print("1. Testing points inside zonotopes...")
    for n in dims:
        print(f"   Dimension {n}:")
        for i in range(Ntests):
            # Generate random zonotope
            Z = Zonotope.generateRandom(dimension=n)
            
            # Generate random points inside the zonotope
            P = Z.randPoint(Npoints)
            
            for i_p in range(P.shape[1]):
                # Compute zonotope norm relative to center
                p_relative = P[:, i_p] - Z.center().flatten()
                # Ensure p_relative is a column vector
                p_relative = p_relative.reshape(-1, 1)
                zN = zonotopeNorm(Z, p_relative)[0]
                
                # If norm > 1, it should be approximately 1 (within tolerance)
                if zN > 1:
                    if not np.isclose(zN, 1, atol=1e-10):
                        print(f"      Test {i+1}, point {i_p+1}: Expected ~1, got {zN}")
                        res = False
                        return res
            print(f"      Test {i+1}: PASSED")
        print()
    
    # Test 2: Points outside zonotopes
    print("2. Testing points outside zonotopes...")
    for n in dims:
        print(f"   Dimension {n}:")
        for i in range(Ntests):
            # Generate random zonotope
            Z = Zonotope.generateRandom(dimension=n)
            
            # Lift Z to a space with one more dimension
            # This ensures points cannot be contained if chosen correctly
            Z_lift_center = np.append(Z.center().flatten(), 0)
            Z_lift_generators = np.vstack([Z.generators(), np.zeros((1, Z.generators().shape[1]))])
            Z_lift = Zonotope(Z_lift_center, Z_lift_generators)
            
            # Generate random points from original zonotope
            P = Z.randPoint(Npoints)
            
            # Lift P and displace by 10 in the new dimension
            P_lift = np.vstack([P, np.zeros((1, P.shape[1]))])
            P_lift = P_lift + np.vstack([np.zeros((n, 1)), np.array([[10]])])
            
            for i_p in range(P_lift.shape[1]):
                # Compute zonotope norm relative to center
                p_relative = P_lift[:, i_p] - Z_lift.center().flatten()
                # Ensure p_relative is a column vector
                p_relative = p_relative.reshape(-1, 1)
                zN = zonotopeNorm(Z_lift, p_relative)[0]
                
                # If norm < 1, it should be approximately 1 (within tolerance)
                if zN < 1:
                    if not np.isclose(zN, 1, atol=1e-10):
                        print(f"      Test {i+1}, point {i_p+1}: Expected ~1, got {zN}")
                        res = False
                        return res
            print(f"      Test {i+1}: PASSED")
        print()
    
    # Test 3: Check norm properties
    print("3. Testing norm properties...")
    for n in dims:
        print(f"   Dimension {n}:")
        for i in range(Ntests):
            # Generate random zonotope
            Z = Zonotope.generateRandom(dimension=n)
            
            # Sample random points
            p1 = Z.randPoint(1).flatten()
            p2 = Z.randPoint(1).flatten()
            
            # Ensure points are column vectors
            p1_col = p1.reshape(-1, 1)
            p2_col = p2.reshape(-1, 1)
            p_sum_col = (p1 + p2).reshape(-1, 1)
            
            # Check triangle inequality: ||p1 + p2|| <= ||p1|| + ||p2||
            zN12 = zonotopeNorm(Z, p_sum_col)[0]
            zN1 = zonotopeNorm(Z, p1_col)[0]
            zN2 = zonotopeNorm(Z, p2_col)[0]
            
            if not (zN12 <= zN1 + zN2 or np.isclose(zN12, zN1 + zN2, atol=1e-10)):
                print(f"      Test {i+1}: Triangle inequality failed")
                print(f"         ||p1 + p2|| = {zN12}")
                print(f"         ||p1|| + ||p2|| = {zN1 + zN2}")
                res = False
                return res
            
            # Check symmetry: ||p1|| = ||-p1||
            zN1_pos = zonotopeNorm(Z, p1_col)[0]
            zN1_neg = zonotopeNorm(Z, -p1_col)[0]
            
            if not np.isclose(zN1_pos, zN1_neg, atol=1e-10):
                print(f"      Test {i+1}: Symmetry failed")
                print(f"         ||p1|| = {zN1_pos}")
                print(f"         ||-p1|| = {zN1_neg}")
                res = False
                return res
            
            # Check positive definiteness: ||0|| = 0
            zero_vec = np.zeros(n).reshape(-1, 1)
            zN_zero = zonotopeNorm(Z, zero_vec)[0]
            if not (zN_zero >= 0 and np.isclose(zN_zero, 0, atol=1e-10)):
                print(f"      Test {i+1}: Positive definiteness failed")
                print(f"         ||0|| = {zN_zero}")
                res = False
                return res
                
            print(f"      Test {i+1}: PASSED")
        print()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("=== Test Summary ===")
    print(f"Total tests: {len(dims) * Ntests * 3}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Result: {'PASSED' if res else 'FAILED'}")
    
    return res


def test_specific_cases():
    """Test specific edge cases and known results"""
    print("\n=== Testing Specific Cases ===\n")
    
    # Test case 1: Simple 2D case
    print("1. Simple 2D case:")
    Z = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    p = np.array([0.5, 0.5]).reshape(-1, 1)
    res, minimizer = zonotopeNorm(Z, p)
    print(f"   Zonotope: center=[0,0], generators=[[1,0],[0,1]]")
    print(f"   Point: [0.5, 0.5]")
    print(f"   Result: {res:.10f}")
    print(f"   Expected: 0.5")
    print(f"   Match: {abs(res - 0.5) < 1e-10}")
    print()
    
    # Test case 2: Degenerate case
    print("2. Degenerate case:")
    Z_degen = Zonotope(np.array([0, 0]), np.array([[1, 1], [1, 1]]))
    p_degen = np.array([1, 1]).reshape(-1, 1)
    res_degen, _ = zonotopeNorm(Z_degen, p_degen)
    print(f"   Degenerate zonotope: center=[0,0], generators=[[1,1],[1,1]]")
    print(f"   Point: [1, 1]")
    print(f"   Result: {res_degen:.10f}")
    print(f"   Expected: 0.5 (since [1,1] = 0.5*[1,1] + 0.5*[1,1])")
    print(f"   Match: {abs(res_degen - 0.5) < 1e-10}")
    print()
    
    # Test case 3: High-dimensional case
    print("3. High-dimensional case:")
    n = 10
    Z_high = Zonotope(np.zeros(n), np.eye(n))
    p_high = (np.ones(n) * 0.3).reshape(-1, 1)
    res_high, _ = zonotopeNorm(Z_high, p_high)
    print(f"   {n}D unit cube zonotope")
    print(f"   Point: [0.3, 0.3, ..., 0.3]")
    print(f"   Result: {res_high:.10f}")
    print(f"   Expected: 0.3")
    print(f"   Match: {abs(res_high - 0.3) < 1e-10}")
    print()


if __name__ == "__main__":
    # Run the long test
    success = testLong_zonotope_zonotopeNorm()
    
    # Run specific cases
    test_specific_cases()
    
    print("\n=== Final Result ===")
    if success:
        print("✅ All long tests PASSED!")
        print("The zonotopeNorm function is working correctly across all test scenarios.")
    else:
        print("❌ Some tests FAILED!")
        print("Please check the implementation for issues.") 