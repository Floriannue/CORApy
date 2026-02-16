"""
Analyze error_adm_horizon growth from trace files
Identifies where growth starts and key transition points
"""

import os
import re
import glob
import numpy as np

def parse_error_adm_horizon(fname):
    """Extract initial error_adm_horizon from trace file"""
    try:
        with open(fname, 'r') as f:
            content = f.read()
            # Find the line with Initial error_adm_horizon
            match = re.search(r'Initial error_adm_horizon:\s*\[\[([\d.e+-]+)\]', content)
            if match:
                val = float(match.group(1))
                step_num = int(re.search(r'step(\d+)', fname).group(1))
                return step_num, val
    except Exception as e:
        pass
    return None, None

def analyze_growth():
    """Analyze error_adm_horizon growth pattern"""
    trace_files = sorted(glob.glob('intermediate_values_step*_inner_loop.txt'), 
                        key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
    
    if not trace_files:
        print("No trace files found!")
        return
    
    print(f"Analyzing {len(trace_files)} trace files...\n")
    
    # Parse all values
    values = []
    for fname in trace_files:
        step, val = parse_error_adm_horizon(fname)
        if step is not None and val is not None:
            values.append((step, val))
    
    if not values:
        print("No error_adm_horizon values found!")
        return
    
    values.sort(key=lambda x: x[0])
    
    # Find growth phases
    print("=== Growth Phase Analysis ===\n")
    
    # Phase 1: Initial (should be stable/decreasing)
    print("Phase 1: Initial Steps (first 50)")
    initial_values = [v[1] for v in values[:50]]
    if initial_values:
        print(f"  Range: {min(initial_values):.6e} to {max(initial_values):.6e}")
        print(f"  Trend: {'Decreasing' if initial_values[-1] < initial_values[0] else 'Increasing'}")
    
    # Phase 2: Find where growth accelerates
    print("\nPhase 2: Growth Acceleration Points")
    growth_rates = []
    for i in range(1, len(values)):
        step1, val1 = values[i-1]
        step2, val2 = values[i]
        if val1 > 0:
            rate = val2 / val1
            growth_rates.append((step2, rate, val1, val2))
    
    # Find significant acceleration points
    significant_growth = [(s, r, v1, v2) for s, r, v1, v2 in growth_rates if r > 1.5]
    
    if significant_growth:
        print(f"  Found {len(significant_growth)} steps with >1.5x growth:")
        for step, rate, v1, v2 in significant_growth[:10]:
            print(f"    Step {step}: {v1:.6e} -> {v2:.6e} ({rate:.2f}x)")
    
    # Phase 3: Find first explosion point
    print("\nPhase 3: First Explosion Point")
    explosion_threshold = 1e6  # Values above this are "exploded"
    first_explosion = None
    for step, val in values:
        if val > explosion_threshold:
            first_explosion = (step, val)
            break
    
    if first_explosion:
        step, val = first_explosion
        print(f"  First step above 1e6: Step {step} with value {val:.6e}")
        # Show context around explosion
        idx = next(i for i, (s, _) in enumerate(values) if s == step)
        print(f"  Context:")
        for i in range(max(0, idx-3), min(len(values), idx+4)):
            s, v = values[i]
            marker = " <-- EXPLOSION" if s == step else ""
            print(f"    Step {s}: {v:.6e}{marker}")
    
    # Phase 4: Final state
    print("\nPhase 4: Final State")
    if values:
        last_step, last_val = values[-1]
        print(f"  Last step: {last_step}")
        print(f"  Last value: {last_val:.6e}")
        print(f"  Total growth: {last_val / values[0][1]:.2e}x")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if first_explosion:
        step, _ = first_explosion
        print(f"1. Focus on Step {step} - first explosion point")
        print(f"2. Compare Step {step-1} vs Step {step} to see what changed")
        print(f"3. Compare with MATLAB at Step {step} to find divergence")
    else:
        print("1. Growth is gradual - compare early steps with MATLAB")
        print("2. Look for first step where Python and MATLAB diverge")
    
    # Identify key steps for comparison
    print("\n=== Key Steps for Comparison ===")
    key_steps = []
    if len(values) > 0:
        key_steps.append(1)  # First step
    if len(values) > 10:
        key_steps.append(10)  # Early step
    if len(values) > 100:
        key_steps.append(100)  # Mid step
    if first_explosion:
        step, _ = first_explosion
        key_steps.append(step - 1)  # Step before explosion
        key_steps.append(step)  # Explosion step
    if len(values) > 500:
        key_steps.append(500)  # Later step
    
    print("Compare these steps between MATLAB and Python:")
    for step in sorted(set(key_steps)):
        if step <= len(values):
            val = next(v for s, v in values if s == step)
            print(f"  Step {step}: error_adm_horizon = {val:.6e}")

if __name__ == '__main__':
    analyze_growth()
