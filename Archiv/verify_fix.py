"""Quick verification that Python code matches MATLAB"""
import re

# Check Python code
with open('cora_python/contDynamics/nonlinearSys/linReach_adaptive.py', 'r') as f:
    python_code = f.read()

# Check MATLAB code  
with open('cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m', 'r') as f:
    matlab_code = f.read()

print("=== Verification: finitehorizon Capping ===")

# Check Python line 78
python_match = re.search(r'min\(params\[.tFinal.\] - options\[.t.\], finitehorizon\)', python_code)
if python_match:
    # Check if it's assigned
    context = python_code[max(0, python_match.start()-50):python_match.end()+50]
    if 'finitehorizon = min(' in context:
        print("[X] Python: finitehorizon IS assigned (WRONG - should match MATLAB)")
    else:
        print("[OK] Python: min() result NOT assigned (CORRECT - matches MATLAB)")
        print(f"   Line: {context.strip()}")
else:
    print("[X] Could not find Python min() call")

# Check MATLAB line 84
matlab_match = re.search(r'min\(\[params\.tFinal - options\.t, finitehorizon\]\);', matlab_code)
if matlab_match:
    context = matlab_code[max(0, matlab_match.start()-50):matlab_match.end()+50]
    if 'finitehorizon = min(' in context:
        print("[X] MATLAB: finitehorizon IS assigned (unexpected)")
    else:
        print("[OK] MATLAB: min() result NOT assigned (as expected)")
        print(f"   Line: {context.strip()}")
else:
    print("[X] Could not find MATLAB min() call")

print("\n=== Conclusion ===")
print("If both show 'NOT assigned', then Python matches MATLAB correctly.")
print("Python should now also abort early like MATLAB.")
