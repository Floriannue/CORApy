"""
Comparison script to compare intermediate values between MATLAB and Python
This reads the trace files and compares key values step by step
"""

import os
import re
import numpy as np
from pathlib import Path

def parse_value_line(line):
    """Parse a line like 'RallError radius_max: 1.234567e+10'"""
    match = re.match(r'(\w+(?:\s+\w+)*):\s*(.+)', line)
    if match:
        key = match.group(1).strip()
        value_str = match.group(2).strip()
        return key, value_str
    return None, None

def parse_array_line(line):
    """Parse a line like 'RallError center: [1.234 5.678]'"""
    match = re.match(r'(\w+(?:\s+\w+)*):\s*\[(.+)\]', line)
    if match:
        key = match.group(1).strip()
        values_str = match.group(2).strip()
        try:
            # Try to parse as space-separated or comma-separated values
            values = [float(v.strip()) for v in re.split(r'[,\s]+', values_str) if v.strip()]
            return key, np.array(values)
        except ValueError:
            return key, values_str
    return None, None

def compare_files(matlab_file, python_file, tolerance=1e-10):
    """Compare MATLAB and Python intermediate value files"""
    
    if not os.path.exists(matlab_file):
        print(f"MATLAB file not found: {matlab_file}")
        return False
    
    if not os.path.exists(python_file):
        print(f"Python file not found: {python_file}")
        return False
    
    print(f"\n=== Comparing {matlab_file} vs {python_file} ===\n")
    
    # Read MATLAB file
    with open(matlab_file, 'r') as f:
        matlab_lines = f.readlines()
    
    # Read Python file
    with open(python_file, 'r') as f:
        python_lines = f.readlines()
    
    # Parse both files
    matlab_values = {}
    python_values = {}
    
    current_section = None
    for line in matlab_lines:
        line = line.strip()
        if line.startswith('---'):
            current_section = line
            continue
        if ':' in line:
            key, value = parse_value_line(line) or parse_array_line(line)
            if key:
                full_key = f"{current_section}:{key}" if current_section else key
                matlab_values[full_key] = value
    
    current_section = None
    for line in python_lines:
        line = line.strip()
        if line.startswith('---'):
            current_section = line
            continue
        if ':' in line:
            key, value = parse_value_line(line) or parse_array_line(line)
            if key:
                full_key = f"{current_section}:{key}" if current_section else key
                python_values[full_key] = value
    
    # Compare values
    all_keys = set(matlab_values.keys()) | set(python_values.keys())
    matches = 0
    mismatches = 0
    missing = 0
    
    for key in sorted(all_keys):
        matlab_val = matlab_values.get(key)
        python_val = python_values.get(key)
        
        if matlab_val is None:
            print(f"⚠️  {key}: Missing in MATLAB, Python has: {python_val}")
            missing += 1
        elif python_val is None:
            print(f"⚠️  {key}: Missing in Python, MATLAB has: {matlab_val}")
            missing += 1
        else:
            # Try to compare as numbers
            try:
                if isinstance(matlab_val, np.ndarray) and isinstance(python_val, np.ndarray):
                    if matlab_val.shape == python_val.shape:
                        diff = np.abs(matlab_val - python_val)
                        max_diff = np.max(diff)
                        if max_diff < tolerance:
                            print(f"✅ {key}: Match (max diff: {max_diff:.2e})")
                            matches += 1
                        else:
                            print(f"❌ {key}: Mismatch (max diff: {max_diff:.2e})")
                            print(f"   MATLAB: {matlab_val}")
                            print(f"   Python: {python_val}")
                            mismatches += 1
                    else:
                        print(f"❌ {key}: Shape mismatch (MATLAB: {matlab_val.shape}, Python: {python_val.shape})")
                        mismatches += 1
                elif isinstance(matlab_val, (int, float)) and isinstance(python_val, (int, float, np.number)):
                    diff = abs(float(matlab_val) - float(python_val))
                    if diff < tolerance:
                        print(f"✅ {key}: Match (diff: {diff:.2e})")
                        matches += 1
                    else:
                        print(f"❌ {key}: Mismatch (diff: {diff:.2e})")
                        print(f"   MATLAB: {matlab_val}")
                        print(f"   Python: {python_val}")
                        mismatches += 1
                else:
                    # String comparison
                    if str(matlab_val) == str(python_val):
                        print(f"✅ {key}: Match")
                        matches += 1
                    else:
                        print(f"❌ {key}: Mismatch")
                        print(f"   MATLAB: {matlab_val}")
                        print(f"   Python: {python_val}")
                        mismatches += 1
            except Exception as e:
                print(f"⚠️  {key}: Comparison error: {e}")
                print(f"   MATLAB: {matlab_val}")
                print(f"   Python: {python_val}")
                mismatches += 1
    
    print(f"\n=== Summary ===")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Missing: {missing}")
    print(f"Total: {len(all_keys)}")
    
    return mismatches == 0 and missing == 0

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_intermediate_values.py <matlab_file> <python_file> [tolerance]")
        print("Example: python compare_intermediate_values.py matlab_intermediate_values.txt python_intermediate_values.txt 1e-10")
        sys.exit(1)
    
    matlab_file = sys.argv[1]
    python_file = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10
    
    success = compare_files(matlab_file, python_file, tolerance)
    sys.exit(0 if success else 1)
