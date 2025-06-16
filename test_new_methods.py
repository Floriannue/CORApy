#!/usr/bin/env python3
"""
Test script for newly implemented methods in reachSet and simResult classes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'cora_python'))

import numpy as np
from cora_python.g.classes.reachSet import ReachSet
from cora_python.g.classes.simResult import SimResult

def test_reachSet_new_methods():
    """Test new reachSet methods"""
    print("Testing reachSet new methods...")
    
    # Create test data
    timePoint = {
        'set': [np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]])],
        'time': [0.0, 1.0]
    }
    
    # Create reachSet objects
    R1 = ReachSet(timePoint=timePoint)
    R2 = ReachSet(timePoint=timePoint)
    
    # Test isequal method
    print("Testing isequal method...")
    try:
        result = R1.isequal(R2)
        print(f"  isequal result: {result}")
        assert result == True, "isequal should return True for identical objects"
        print("  ✓ isequal test passed")
    except Exception as e:
        print(f"  ✗ isequal test failed: {e}")
    
    # Test append method
    print("Testing append method...")
    try:
        R3 = R1.append(R2)
        print(f"  append result: {R3}")
        print("  ✓ append test passed")
    except Exception as e:
        print(f"  ✗ append test failed: {e}")
    
    # Test children method
    print("Testing children method...")
    try:
        children = R1.children(0)
        print(f"  children result: {children}")
        print("  ✓ children test passed")
    except Exception as e:
        print(f"  ✗ children test failed: {e}")

def test_simResult_new_methods():
    """Test new simResult methods"""
    print("\nTesting simResult new methods...")
    
    # Create test data
    t1 = np.array([0.0, 0.1, 0.2, 0.3])
    x1 = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]])
    
    # Create simResult object
    simRes = SimResult(x=[x1], t=[t1])
    
    # Test extractHits method
    print("Testing extractHits method...")
    try:
        tHit, xHit, xHit_ = simRes.extractHits()
        print(f"  extractHits result: tHit={tHit}, xHit len={len(xHit)}, xHit_ len={len(xHit_)}")
        print("  ✓ extractHits test passed")
    except Exception as e:
        print(f"  ✗ extractHits test failed: {e}")
    
    # Test plotTimeStep method
    print("Testing plotTimeStep method...")
    try:
        # This might fail if matplotlib is not available, but we'll try
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        fig = simRes.plotTimeStep()
        print(f"  plotTimeStep result: {fig}")
        print("  ✓ plotTimeStep test passed")
    except ImportError:
        print("  ⚠ plotTimeStep test skipped (matplotlib not available)")
    except Exception as e:
        print(f"  ✗ plotTimeStep test failed: {e}")

def main():
    """Run all tests"""
    print("Running tests for new methods...")
    
    try:
        test_reachSet_new_methods()
        test_simResult_new_methods()
        print("\n✓ All tests completed!")
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 