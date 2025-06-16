#!/usr/bin/env python3
"""
Simple test for reachSet and simResult classes
"""

import sys
import os
import numpy as np

# Add the path to the classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python', 'g', 'classes'))

def test_reachSet():
    """Test reachSet class"""
    print("Testing reachSet class...")
    
    try:
        # Import directly from the reachSet module
        from reachSet.reachSet import ReachSet
        
        # Create test data
        timePoint = {
            'set': [np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]])],
            'time': [0.0, 1.0]
        }
        
        # Create reachSet objects
        R1 = ReachSet(timePoint=timePoint)
        R2 = ReachSet(timePoint=timePoint)
        
        print(f"  Created reachSet: {R1}")
        
        # Test basic functionality
        print("  Testing isemptyobject...")
        empty = R1.isemptyobject()
        print(f"    isemptyobject result: {empty}")
        
        print("  ✓ reachSet basic test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ reachSet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simResult():
    """Test simResult class"""
    print("\nTesting simResult class...")
    
    try:
        # Import directly from the simResult module
        from simResult.simResult import SimResult
        
        # Create test data
        t1 = np.array([0.0, 0.1, 0.2, 0.3])
        x1 = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]])
        
        # Create simResult object
        simRes = SimResult(x=[x1], t=[t1])
        
        print(f"  Created simResult: {simRes}")
        
        # Test basic functionality
        print("  Testing isemptyobject...")
        empty = simRes.isemptyobject()
        print(f"    isemptyobject result: {empty}")
        
        print("  ✓ simResult basic test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ simResult test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Running simple tests...")
    
    success = True
    success &= test_reachSet()
    success &= test_simResult()
    
    if success:
        print("\n✓ All basic tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 