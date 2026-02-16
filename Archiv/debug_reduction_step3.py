"""debug_reduction_step3 - Debug reduction for Step 3 specifically"""

import numpy as np
import pickle
import scipy.io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.contSet.zonotope import Zonotope

print("=" * 80)
print("DEBUGGING REDUCTION FOR STEP 3")
print("=" * 80)

# Load Python log to get Z before reduction
python_file = 'upstream_python_log.pkl'
with open(python_file, 'rb') as f:
    python_data = pickle.load(f)
python_upstream = python_data.get('upstreamLog', [])

# Find Step 3 entry
step3_entries = [e for e in python_upstream if e.get('step') == 3]
if step3_entries:
    entry = step3_entries[-1]  # Last one (converged)
    
    Z_before = entry.get('Z_before_quadmap')
    if Z_before:
        center = np.asarray(Z_before['center'])
        generators = np.asarray(Z_before['generators'])
        
        print(f"\nZ before quadMap (after reduction):")
        print(f"  Center shape: {center.shape}")
        print(f"  Generators shape: {generators.shape}")
        print(f"  Number of generators: {generators.shape[1] if len(generators.shape) > 1 else 0}")
        
        # Reconstruct Z
        Z = Zonotope(center, generators)
        
        # Test reduction with sqrt(redFactor)
        # We need to find what redFactor is
        # From the code: reduce(R,'adaptive',sqrt(options.redFactor))
        # Typical redFactor values are around 0.1-0.3
        
        print(f"\nTesting reduction with different diagpercent values:")
        print(f"(diagpercent = sqrt(redFactor), so if redFactor=0.1, diagpercent=0.316)")
        
        diagpercent_values = [0.1, 0.2, 0.3, 0.316, 0.4, 0.5]
        
        for diagpercent in diagpercent_values:
            try:
                # We need to reconstruct the original Z before reduction
                # But we don't have it. Let's check if we can infer it
                print(f"\n  Testing diagpercent = {diagpercent:.3f}")
                print(f"    Current Z has {generators.shape[1]} generators")
                print(f"    (Cannot test without original Z before reduction)")
            except Exception as e:
                print(f"    ERROR: {e}")

print("\n" + "=" * 80)
print("We need to track R before reduction to compare")
print("=" * 80)
