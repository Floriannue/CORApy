"""Trace the exact logic flow for timeStepequalHorizon condition"""
import pickle
import numpy as np

print("=" * 80)
print("TRACING timeStepequalHorizon LOGIC STEP BY STEP")
print("=" * 80)

# The logic flow should be:
# 1. Step 2 starts: timeStep = finitehorizon (line 98)
# 2. Run 1: timeStep = _aux_optimaldeltat(...) (line 599-601)
# 3. Run 1: timeStep = min(timeStep, tFinal - t) (line 602) - ONLY in Python!
# 4. After Run 1: check if timeStep == finitehorizon (line 677)

print("\nKEY DIFFERENCE FOUND:")
print("=" * 80)
print("Python line 602: options['timeStep'] = min(options['timeStep'], params['tFinal'] - options['t'])")
print("MATLAB line 595: options.timeStep = min([options.timeStep, params.tFinal - options.t])")
print("\nBoth do the same min() operation, so that's not the issue.")
print("\nThe real question: When does _aux_optimaldeltat return exactly finitehorizon?")
print("If it returns a different value, then timeStep != finitehorizon and the check fails.")

# Check optimaldeltat log to see what it returns
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

optimaldeltat_log = python_log.get('optimaldeltatLog', [])

# Find Step 2 Run 1 optimaldeltat call
step2_run1_opt = None
for entry in optimaldeltat_log:
    if isinstance(entry, dict) and entry.get('step') == 2:
        step2_run1_opt = entry
        break

if step2_run1_opt:
    print(f"\nStep 2 Run 1 optimaldeltat:")
    deltat = step2_run1_opt.get('deltat')  # This is finitehorizon
    deltatest = step2_run1_opt.get('deltatest')  # This is what _aux_optimaldeltat returns
    print(f"  Input deltat (finitehorizon): {deltat}")
    print(f"  Output deltatest (timeStep): {deltatest}")
    if deltat and deltatest:
        diff = abs(deltat - deltatest)
        rel_diff = diff / abs(deltat) if deltat != 0 else diff
        print(f"  Absolute difference: {diff:.15e}")
        print(f"  Relative difference: {rel_diff:.15e}")
        print(f"  Are they equal? {deltat == deltatest}")
        print(f"  Within 1e-10? {diff < 1e-10}")
        print(f"  Within 1e-6? {diff < 1e-6}")
        
        if deltat != deltatest:
            print(f"\n  [ISSUE] _aux_optimaldeltat returns {deltatest} instead of {deltat}")
            print(f"  This means timeStep != finitehorizon, so the check fails.")
            print(f"  Need to understand why MATLAB's check succeeds but Python's doesn't.")

# Also check Step 1 to see the pattern
step1_opt = []
for entry in optimaldeltat_log:
    if isinstance(entry, dict) and entry.get('step') == 1:
        step1_opt.append(entry)

if step1_opt:
    print(f"\nStep 1 optimaldeltat calls:")
    for entry in step1_opt[:3]:  # First 3
        deltat = entry.get('deltat')
        deltatest = entry.get('deltatest')
        if deltat and deltatest:
            print(f"  deltat={deltat:.12e}, deltatest={deltatest:.12e}, diff={abs(deltat-deltatest):.15e}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("The check happens AFTER _aux_optimaldeltat modifies timeStep.")
print("If _aux_optimaldeltat returns a value different from finitehorizon,")
print("then timeStep != finitehorizon and timeStepequalHorizon remains False.")
print("\nNeed to check:")
print("1. Does MATLAB's aux_optimaldeltat return exactly finitehorizon in Step 2 Run 1?")
print("2. Does Python's _aux_optimaldeltat return exactly finitehorizon?")
print("3. If not, why does MATLAB's check succeed but Python's doesn't?")
