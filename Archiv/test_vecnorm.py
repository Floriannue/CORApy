"""Test vecnorm computation"""
import numpy as np

# Create a test matrix similar to gensdiag
# gensdiag is (n, nrG_red) where each column is a vector
n = 2
nrG_red = 3
gensdiag = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])

print("Test matrix gensdiag:")
print(gensdiag)
print(f"Shape: {gensdiag.shape}")

# MATLAB: h = 2 * vecnorm(gensdiag,2);
# vecnorm(gensdiag,2) without dimension argument defaults to dimension 1 (columns)
# This computes sqrt(sum(gensdiag.^2,1)) for each column

# Manual computation (MATLAB equivalent)
h_manual = 2 * np.sqrt(np.sum(gensdiag**2, axis=0))
print(f"\nManual computation (sqrt(sum(gensdiag^2, axis=0))):")
print(h_manual)

# Python: h = 2 * np.linalg.norm(gensdiag, axis=0, ord=2)
h_python = 2 * np.linalg.norm(gensdiag, axis=0, ord=2)
print(f"\nPython np.linalg.norm(gensdiag, axis=0, ord=2):")
print(h_python)

print(f"\nMatch: {np.allclose(h_manual, h_python)}")

# Test with different shapes
print("\n" + "="*80)
print("Testing with different shapes")
print("="*80)

for n in [2, 3, 5]:
    for nrG_red in [2, 3, 5]:
        gensdiag = np.random.rand(n, nrG_red)
        h_manual = 2 * np.sqrt(np.sum(gensdiag**2, axis=0))
        h_python = 2 * np.linalg.norm(gensdiag, axis=0, ord=2)
        match = np.allclose(h_manual, h_python)
        print(f"n={n}, nrG_red={nrG_red}: Match={match}")
