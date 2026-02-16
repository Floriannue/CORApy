"""analyze_timestep_shrinkage - Analyze when and why time steps start shrinking"""

import numpy as np
import pickle

# Load Python key values
with open('jetEngine_python_key_values.pkl', 'rb') as f:
    python_data = pickle.load(f)

print("=" * 80)
print("TIMESTEP SHRINKAGE ANALYSIS")
print("=" * 80)

if 'stepsize' in python_data and len(python_data['stepsize']) > 0:
    stepsize = python_data['stepsize']
    
    print(f"\nTotal steps: {len(stepsize)}")
    print(f"First 10 stepsize: {[f'{x:.6e}' for x in stepsize[:10]]}")
    print(f"Last 10 stepsize: {[f'{x:.6e}' for x in stepsize[-10:]]}")
    
    # Find when stepsize starts shrinking
    print("\nAnalyzing stepsize progression:")
    shrinking_regions = []
    for i in range(1, len(stepsize)):
        if stepsize[i] < stepsize[i-1] * 0.9:  # 10% decrease
            shrinking_regions.append((i+1, stepsize[i-1], stepsize[i], stepsize[i]/stepsize[i-1]))
    
    if shrinking_regions:
        print(f"\nFound {len(shrinking_regions)} regions where stepsize shrinks by >10%:")
        for step, prev, curr, ratio in shrinking_regions[:10]:
            print(f"  Step {step}: {prev:.6e} -> {curr:.6e} (ratio: {ratio:.6f})")
    
    # Find when stepsize becomes very small (< 1e-6)
    very_small_idx = [i for i, x in enumerate(stepsize) if x < 1e-6]
    if very_small_idx:
        first_very_small = very_small_idx[0]
        print(f"\nFirst very small stepsize (< 1e-6) at step {first_very_small + 1}")
        print(f"  Value: {stepsize[first_very_small]:.6e}")
        if first_very_small > 0:
            print(f"  Previous: {stepsize[first_very_small - 1]:.6e}")
            print(f"  Ratio: {stepsize[first_very_small] / stepsize[first_very_small - 1]:.6e}")
    
    # Analyze finitehorizon growth
    if 'debug_finitehorizon' in python_data and len(python_data['debug_finitehorizon']) > 0:
        print("\n" + "=" * 80)
        print("FINITEHORIZON GROWTH ANALYSIS")
        print("=" * 80)
        
        # Find when finitehorizon becomes unbounded
        unbounded_steps = []
        for entry in python_data['debug_finitehorizon']:
            if entry['computed_finitehorizon'] > entry['remTime']:
                unbounded_steps.append(entry)
        
        if unbounded_steps:
            print(f"\nFound {len(unbounded_steps)} steps where finitehorizon > remTime (unbounded):")
            for entry in unbounded_steps[:10]:
                print(f"\n  Step {entry['step']}:")
                print(f"    computed_finitehorizon: {entry['computed_finitehorizon']:.6e}")
                print(f"    remTime: {entry['remTime']:.6f}")
                print(f"    ratio: {entry['computed_finitehorizon'] / entry['remTime']:.2e}")
                print(f"    prev_varphi: {entry['prev_varphi']:.6f}")
                print(f"    zetaphi: {entry['zetaphi']:.6f}")
        else:
            print("\nNo unbounded finitehorizon found in tracked steps")
            print("Checking if finitehorizon grows over time...")
            
            # Check growth trend
            if len(python_data['debug_finitehorizon']) > 10:
                early_avg = np.mean([e['computed_finitehorizon'] for e in python_data['debug_finitehorizon'][:10]])
                late_avg = np.mean([e['computed_finitehorizon'] for e in python_data['debug_finitehorizon'][-10:]])
                print(f"  Early average (first 10): {early_avg:.6e}")
                print(f"  Late average (last 10): {late_avg:.6e}")
                print(f"  Growth factor: {late_avg / early_avg:.6f}")
