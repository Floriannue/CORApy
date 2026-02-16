"""Trace trueError computation to find source of differences"""
import re
import numpy as np

def extract_values_from_trace(fname):
    """Extract all relevant values from trace file"""
    with open(fname, 'r') as f:
        content = f.read()
    
    values = {}
    
    # Extract VerrorDyn center and radius - handle both MATLAB (semicolon) and Python (space) formats
    verror_center_match = re.search(r'VerrorDyn center:\s*\[(.*?)\]', content, re.DOTALL)
    if verror_center_match:
        center_str = verror_center_match.group(1).replace(';', ',').replace('\n', ' ')
        # Handle Python format with spaces between numbers
        center_str = re.sub(r'\s+', ' ', center_str)
        try:
            # Split by comma or space
            parts = re.split(r'[,;\s]+', center_str)
            center_vals = [float(x.strip()) for x in parts if x.strip() and x.strip() != '[' and x.strip() != ']']
            if center_vals:
                values['VerrorDyn_center'] = np.array(center_vals)
        except Exception as e:
            print(f"Error parsing VerrorDyn center: {e}")
            print(f"  String: {center_str[:200]}")
    
    verror_radius_match = re.search(r'VerrorDyn radius:\s*\[(.*?)\]', content, re.DOTALL)
    if verror_radius_match:
        radius_str = verror_radius_match.group(1).replace(';', ',').replace('\n', ' ')
        radius_str = re.sub(r'\s+', ' ', radius_str)
        try:
            parts = re.split(r'[,;\s]+', radius_str)
            radius_vals = [float(x.strip()) for x in parts if x.strip() and x.strip() != '[' and x.strip() != ']']
            if radius_vals:
                values['VerrorDyn_radius'] = np.array(radius_vals)
        except Exception as e:
            print(f"Error parsing VerrorDyn radius: {e}")
            print(f"  String: {radius_str[:200]}")
    
    # Extract trueError - handle both formats
    trueerror_match = re.search(r'trueError:\s*\[(.*?)\]', content, re.DOTALL)
    if trueerror_match:
        trueerror_str = trueerror_match.group(1).replace(';', ',').replace('\n', ' ')
        trueerror_str = re.sub(r'\s+', ' ', trueerror_str)
        try:
            parts = re.split(r'[,;\s]+', trueerror_str)
            trueerror_vals = [float(x.strip()) for x in parts if x.strip() and x.strip() != '[' and x.strip() != ']']
            if trueerror_vals:
                values['trueError'] = np.array(trueerror_vals)
        except Exception as e:
            print(f"Error parsing trueError: {e}")
            print(f"  String: {trueerror_str[:200]}")
    
    # Extract errorSec
    errorsec_center_match = re.search(r'errorSec \(after quadMap\) center:\s*\[(.*?)\]', content, re.DOTALL)
    if errorsec_center_match:
        center_str = errorsec_center_match.group(1).replace(';', ',').replace('\n', ' ')
        try:
            center_vals = [float(x.strip()) for x in center_str.split(',') if x.strip()]
            values['errorSec_center'] = np.array(center_vals)
        except:
            pass
    
    errorsec_radius_match = re.search(r'errorSec \(after quadMap\) radius:\s*\[(.*?)\]', content, re.DOTALL)
    if errorsec_radius_match:
        radius_str = errorsec_radius_match.group(1).replace(';', ',').replace('\n', ' ')
        try:
            radius_vals = [float(x.strip()) for x in radius_str.split(',') if x.strip()]
            values['errorSec_radius'] = np.array(radius_vals)
        except:
            pass
    
    # Extract Z before quadMap
    z_center_match = re.search(r'Z \(before quadMap\) center:\s*\[(.*?)\]', content, re.DOTALL)
    if z_center_match:
        center_str = z_center_match.group(1).replace(';', ',').replace('\n', ' ')
        try:
            center_vals = [float(x.strip()) for x in center_str.split(',') if x.strip()]
            values['Z_center'] = np.array(center_vals)
        except:
            pass
    
    z_radius_match = re.search(r'Z \(before quadMap\) radius:\s*\[(.*?)\]', content, re.DOTALL)
    if z_radius_match:
        radius_str = z_radius_match.group(1).replace(';', ',').replace('\n', ' ')
        try:
            radius_vals = [float(x.strip()) for x in radius_str.split(',') if x.strip()]
            values['Z_radius'] = np.array(radius_vals)
        except:
            pass
    
    return values

# Compare step 1
print("=== Tracing trueError Computation for Step 1 ===\n")

matlab_vals = extract_values_from_trace('intermediate_values_step1_inner_loop_matlab.txt')
python_vals = extract_values_from_trace('intermediate_values_step1_inner_loop.txt')

print("MATLAB Values:")
if 'Z_center' in matlab_vals:
    print(f"  Z center: {matlab_vals['Z_center']}")
    print(f"  Z center max abs: {np.max(np.abs(matlab_vals['Z_center'])):.15e}")
if 'Z_radius' in matlab_vals:
    print(f"  Z radius: {matlab_vals['Z_radius']}")
    print(f"  Z radius max: {np.max(matlab_vals['Z_radius']):.15e}")

if 'errorSec_center' in matlab_vals:
    print(f"  errorSec center: {matlab_vals['errorSec_center']}")
    print(f"  errorSec center max abs: {np.max(np.abs(matlab_vals['errorSec_center'])):.15e}")
if 'errorSec_radius' in matlab_vals:
    print(f"  errorSec radius: {matlab_vals['errorSec_radius']}")
    print(f"  errorSec radius max: {np.max(matlab_vals['errorSec_radius']):.15e}")

if 'VerrorDyn_center' in matlab_vals:
    print(f"  VerrorDyn center: {matlab_vals['VerrorDyn_center']}")
    print(f"  VerrorDyn center max abs: {np.max(np.abs(matlab_vals['VerrorDyn_center'])):.15e}")
if 'VerrorDyn_radius' in matlab_vals:
    print(f"  VerrorDyn radius: {matlab_vals['VerrorDyn_radius']}")
    print(f"  VerrorDyn radius max: {np.max(matlab_vals['VerrorDyn_radius']):.15e}")

if 'trueError' in matlab_vals:
    print(f"  trueError: {matlab_vals['trueError']}")
    print(f"  trueError max: {np.max(matlab_vals['trueError']):.15e}")

print("\nPython Values:")
if 'Z_center' in python_vals:
    print(f"  Z center: {python_vals['Z_center']}")
    print(f"  Z center max abs: {np.max(np.abs(python_vals['Z_center'])):.15e}")
if 'Z_radius' in python_vals:
    print(f"  Z radius: {python_vals['Z_radius']}")
    print(f"  Z radius max: {np.max(python_vals['Z_radius']):.15e}")

if 'errorSec_center' in python_vals:
    print(f"  errorSec center: {python_vals['errorSec_center']}")
    print(f"  errorSec center max abs: {np.max(np.abs(python_vals['errorSec_center'])):.15e}")
if 'errorSec_radius' in python_vals:
    print(f"  errorSec radius: {python_vals['errorSec_radius']}")
    print(f"  errorSec radius max: {np.max(python_vals['errorSec_radius']):.15e}")

if 'VerrorDyn_center' in python_vals:
    print(f"  VerrorDyn center: {python_vals['VerrorDyn_center']}")
    print(f"  VerrorDyn center max abs: {np.max(np.abs(python_vals['VerrorDyn_center'])):.15e}")
if 'VerrorDyn_radius' in python_vals:
    print(f"  VerrorDyn radius: {python_vals['VerrorDyn_radius']}")
    print(f"  VerrorDyn radius max: {np.max(python_vals['VerrorDyn_radius']):.15e}")

if 'trueError' in python_vals:
    print(f"  trueError: {python_vals['trueError']}")
    print(f"  trueError max: {np.max(python_vals['trueError']):.15e}")

print("\n=== Differences ===")
if 'VerrorDyn_center' in matlab_vals and 'VerrorDyn_center' in python_vals:
    diff_center = matlab_vals['VerrorDyn_center'] - python_vals['VerrorDyn_center']
    print(f"VerrorDyn center diff: {diff_center}")
    print(f"VerrorDyn center max abs diff: {np.max(np.abs(diff_center)):.15e}")

if 'VerrorDyn_radius' in matlab_vals and 'VerrorDyn_radius' in python_vals:
    diff_radius = matlab_vals['VerrorDyn_radius'] - python_vals['VerrorDyn_radius']
    print(f"VerrorDyn radius diff: {diff_radius}")
    print(f"VerrorDyn radius max abs diff: {np.max(np.abs(diff_radius)):.15e}")

if 'trueError' in matlab_vals and 'trueError' in python_vals:
    diff_trueerror = matlab_vals['trueError'] - python_vals['trueError']
    print(f"trueError diff: {diff_trueerror}")
    print(f"trueError max abs diff: {np.max(np.abs(diff_trueerror)):.15e}")
    print(f"trueError max rel diff: {np.max(np.abs(diff_trueerror)) / np.max(np.abs(matlab_vals['trueError'])):.15e}")

if 'errorSec_center' in matlab_vals and 'errorSec_center' in python_vals:
    diff_errorsec_center = matlab_vals['errorSec_center'] - python_vals['errorSec_center']
    print(f"\n=== errorSec Differences (from quadMap) ===")
    print(f"errorSec center diff: {diff_errorsec_center}")
    print(f"errorSec center max abs diff: {np.max(np.abs(diff_errorsec_center)):.15e}")
    print(f"errorSec center max rel diff: {np.max(np.abs(diff_errorsec_center)) / np.max(np.abs(matlab_vals['errorSec_center'])):.15e}")

if 'errorSec_radius' in matlab_vals and 'errorSec_radius' in python_vals:
    diff_errorsec_radius = matlab_vals['errorSec_radius'] - python_vals['errorSec_radius']
    print(f"errorSec radius diff: {diff_errorsec_radius}")
    print(f"errorSec radius max abs diff: {np.max(np.abs(diff_errorsec_radius)):.15e}")
    print(f"errorSec radius max rel diff: {np.max(np.abs(diff_errorsec_radius)) / np.max(np.abs(matlab_vals['errorSec_radius'])):.15e}")

if 'Z_center' in matlab_vals and 'Z_center' in python_vals:
    diff_z_center = matlab_vals['Z_center'] - python_vals['Z_center']
    print(f"\n=== Z (before quadMap) Differences ===")
    print(f"Z center diff: {diff_z_center}")
    print(f"Z center max abs diff: {np.max(np.abs(diff_z_center)):.15e}")
    if len(matlab_vals['Z_center']) == len(python_vals['Z_center']):
        print(f"Z center max rel diff: {np.max(np.abs(diff_z_center)) / np.max(np.abs(matlab_vals['Z_center'])):.15e}")

if 'Z_radius' in matlab_vals and 'Z_radius' in python_vals:
    diff_z_radius = matlab_vals['Z_radius'] - python_vals['Z_radius']
    print(f"Z radius diff: {diff_z_radius}")
    print(f"Z radius max abs diff: {np.max(np.abs(diff_z_radius)):.15e}")
    if len(matlab_vals['Z_radius']) == len(python_vals['Z_radius']):
        print(f"Z radius max rel diff: {np.max(np.abs(diff_z_radius)) / np.max(np.abs(matlab_vals['Z_radius'])):.15e}")

# Compute trueError from VerrorDyn to verify formula
if 'VerrorDyn_center' in matlab_vals and 'VerrorDyn_radius' in matlab_vals:
    computed_trueerror_matlab = np.abs(matlab_vals['VerrorDyn_center']) + matlab_vals['VerrorDyn_radius']
    print(f"\n=== Verification: trueError = abs(center) + radius ===")
    print(f"MATLAB computed trueError: {computed_trueerror_matlab}")
    print(f"MATLAB actual trueError: {matlab_vals.get('trueError', 'N/A')}")
    if 'trueError' in matlab_vals:
        diff = computed_trueerror_matlab - matlab_vals['trueError']
        print(f"MATLAB diff: {diff}")
        print(f"MATLAB max abs diff: {np.max(np.abs(diff)):.15e}")

if 'VerrorDyn_center' in python_vals and 'VerrorDyn_radius' in python_vals:
    computed_trueerror_python = np.abs(python_vals['VerrorDyn_center']) + python_vals['VerrorDyn_radius']
    print(f"\nPython computed trueError: {computed_trueerror_python}")
    print(f"Python actual trueError: {python_vals.get('trueError', 'N/A')}")
    if 'trueError' in python_vals:
        diff = computed_trueerror_python - python_vals['trueError']
        print(f"Python diff: {diff}")
        print(f"Python max abs diff: {np.max(np.abs(diff)):.15e}")
