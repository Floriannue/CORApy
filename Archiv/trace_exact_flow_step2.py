"""Trace the exact flow for Step 2 to understand when timeStepequalHorizon is set"""
import pickle
import numpy as np

print("=" * 80)
print("TRACING EXACT FLOW FOR STEP 2")
print("=" * 80)

# The flow should be:
# 1. Step 2 starts: timeStep = finitehorizon (line 98)
# 2. Run 1: 
#    - timeStep = _aux_optimaldeltat(...) (line 599-601)
#    - timeStep = min(timeStep, tFinal - t) (line 602) - ONLY for i > 1
# 3. After Run 1: run += 1 (line 675)
# 4. Check: if timeStep == finitehorizon (line 682)
# 5. If true: timeStepequalHorizon = True, timeStep = timeStep * decrFactor
# 6. Run 2: if timeStepequalHorizon, use Rtp_h and Rerror_h

print("\nKEY OBSERVATION:")
print("=" * 80)
print("The check happens AFTER:")
print("  1. _aux_optimaldeltat modifies timeStep")
print("  2. min(timeStep, tFinal - t) potentially modifies timeStep")
print("\nSo timeStep will only equal finitehorizon if:")
print("  1. _aux_optimaldeltat returns exactly finitehorizon (bestIdxnew = 0)")
print("  2. AND min(timeStep, tFinal - t) == finitehorizon")

# Check if tFinal - t might be capping timeStep
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

python_upstream = python_log['upstreamLog']

# Find Step 2 Run 1
step2_run1 = None
for entry in python_upstream:
    if isinstance(entry, dict) and entry.get('step') == 2 and entry.get('run') == 1:
        step2_run1 = entry
        break

if step2_run1:
    print(f"\nStep 2 Run 1:")
    if 'initReach_tracking' in step2_run1:
        it = step2_run1['initReach_tracking']
        timeStep = it.get('timeStep')
        if timeStep:
            print(f"  timeStep after Run 1: {timeStep}")

# Check optimaldeltat to see what it returned
optimaldeltat_log = python_log.get('optimaldeltatLog', [])
step2_opt = None
for entry in optimaldeltat_log:
    if isinstance(entry, dict) and entry.get('step') == 2:
        step2_opt = entry
        break

if step2_opt:
    deltat = step2_opt.get('deltat')  # finitehorizon
    deltatest = step2_opt.get('deltatest')  # what _aux_optimaldeltat returned
    print(f"\n  _aux_optimaldeltat:")
    print(f"    Input (finitehorizon): {deltat}")
    print(f"    Output (deltatest): {deltatest}")
    print(f"    Difference: {abs(deltat - deltatest) if deltat and deltatest else 'N/A'}")
    
    # Check if min() would cap it
    # tFinal = 8.0, need to estimate t at Step 2
    # Step 1 timeStep ~ 0.007, so t at Step 2 start ~ 0.007
    t_approx = 0.007
    tFinal = 8.0
    remTime = tFinal - t_approx
    print(f"\n  min() check:")
    print(f"    Estimated t at Step 2: ~{t_approx}")
    print(f"    tFinal - t: ~{remTime}")
    print(f"    deltatest: {deltatest}")
    print(f"    min(deltatest, remTime): {min(deltatest, remTime) if deltatest else 'N/A'}")
    if deltatest and deltatest < remTime:
        print(f"    [OK] min() does not cap deltatest")
    else:
        print(f"    [ISSUE] min() might cap deltatest")

print("\n" + "=" * 80)
print("THE REAL ISSUE")
print("=" * 80)
print("_aux_optimaldeltat returns a different value than finitehorizon.")
print("This is CORRECT behavior - the optimization found a better time step.")
print("The check 'if timeStep == finitehorizon' correctly fails.")
print("\nBut MATLAB Step 2 Run 2 uses timeStepequalHorizon path.")
print("This means MATLAB's _aux_optimaldeltat must return exactly finitehorizon,")
print("OR there's a different logic path in MATLAB.")
print("\nNeed to check if MATLAB's aux_optimaldeltat returns exactly finitehorizon for Step 2 Run 1.")
