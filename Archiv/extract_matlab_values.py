"""Script to extract MATLAB values from backprop_test_results.mat and update Python test"""
import numpy as np
import scipy.io
import os

# Try to load MATLAB results
mat_file = 'backprop_test_results.mat'
if os.path.exists(mat_file):
    print(f"Loading {mat_file}...")
    data = scipy.io.loadmat(mat_file)
    
    # Extract values (MATLAB saves with variable names as keys)
    c_out = data.get('c_out', None)
    G_out = data.get('G_out', None)
    gc_out = data.get('gc_out', None)
    gG_out = data.get('gG_out', None)
    m = data.get('m', None)
    
    print("\n=== MATLAB Output Values ===")
    print(f"\nc_out shape: {c_out.shape if c_out is not None else 'N/A'}")
    if c_out is not None:
        print(f"c_out:\n{c_out}")
        print(f"c_out (Python format):\n{np.array2string(c_out.flatten(), separator=', ')}")
    
    print(f"\nG_out shape: {G_out.shape if G_out is not None else 'N/A'}")
    if G_out is not None:
        print(f"G_out:\n{G_out}")
        print(f"G_out (Python format):\n{np.array2string(G_out.flatten(), separator=', ')}")
    
    print(f"\ngc_out shape: {gc_out.shape if gc_out is not None else 'N/A'}")
    if gc_out is not None:
        print(f"gc_out:\n{gc_out}")
        print(f"gc_out (Python format):\n{np.array2string(gc_out.flatten(), separator=', ')}")
    
    print(f"\ngG_out shape: {gG_out.shape if gG_out is not None else 'N/A'}")
    if gG_out is not None:
        print(f"gG_out:\n{gG_out}")
        print(f"gG_out (Python format):\n{np.array2string(gG_out.flatten(), separator=', ')}")
    
    print(f"\nm (slope) shape: {m.shape if m is not None else 'N/A'}")
    if m is not None:
        print(f"m:\n{m}")
        print(f"m (Python format):\n{np.array2string(m.flatten(), separator=', ')}")
    
    # Generate Python test code
    print("\n=== Python Test Code ===")
    print("\n# Expected values from MATLAB:")
    if c_out is not None:
        print(f"expected_c_out = np.array({np.array2string(c_out, separator=', ', max_line_width=1000)})")
    if G_out is not None:
        print(f"expected_G_out = np.array({np.array2string(G_out, separator=', ', max_line_width=1000)})")
    if gc_out is not None:
        print(f"expected_gc_out = np.array({np.array2string(gc_out, separator=', ', max_line_width=1000)})")
    if gG_out is not None:
        print(f"expected_gG_out = np.array({np.array2string(gG_out, separator=', ', max_line_width=1000)})")
    if m is not None:
        print(f"expected_m = np.array({np.array2string(m, separator=', ', max_line_width=1000)})")
    
else:
    print(f"{mat_file} not found.")
    print("\nTo generate the MATLAB values:")
    print("1. Open MATLAB")
    print("2. Navigate to this directory")
    print("3. Run: debug_backprop_matlab.m")
    print("4. This will create backprop_test_results.mat")
    print("5. Then run this script again to extract the values")

