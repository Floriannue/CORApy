"""verify_fix_applied - Verify the flatten order fix is in the code"""

import re

# Read the quadMap.py file
with open('cora_python/contSet/zonotope/quadMap.py', 'r') as f:
    content = f.read()

print("=" * 80)
print("VERIFYING FLATTEN ORDER FIX")
print("=" * 80)

# Check for the fix
if "flatten(order='F')" in content:
    print("\n[OK] Fix found: Using column-major flattening (order='F')")
    
    # Find the lines
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "flatten(order='F')" in line:
            print(f"\nLine {i+1}: {line.strip()}")
            # Show context
            if i > 0:
                print(f"Line {i}: {lines[i-1].strip()}")
            if i < len(lines) - 1:
                print(f"Line {i+2}: {lines[i+1].strip()}")
else:
    print("\n[ERROR] Fix NOT found: Should use flatten(order='F')")

# Check for old code (row-major)
if "quadMatoffdiag_flat = quadMatoffdiag.flatten()" in content and "order='F'" not in content:
    print("\n[WARNING] Found old code without order='F'")
elif "quadMatoffdiag_flat = quadMatoffdiag.flatten(order='F')" in content:
    print("\n[OK] quadMatoffdiag flattening uses order='F'")
else:
    print("\n[INFO] Could not find quadMatoffdiag flattening line")

# Check for kInd flattening
if "kInd.flatten(order='F')" in content:
    print("[OK] kInd flattening uses order='F'")
elif "kInd.flatten()" in content and "order='F'" not in content:
    print("[WARNING] kInd flattening does NOT use order='F'")
else:
    print("[INFO] Could not find kInd flattening line")

print("\n" + "=" * 80)
