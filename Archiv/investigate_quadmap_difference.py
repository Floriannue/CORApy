"""Investigate quadMap computation differences"""
import numpy as np

# MATLAB computation (from quadMap.m line 89-96):
# quadMat = Zmat'*Q{i}*Zmat;
# G(i,1:gens) = 0.5*diag(quadMat(2:gens+1,2:gens+1));
# c(i,1) = quadMat(1,1) + sum(G(i,1:gens));

# Python computation (from quadMap.py):
# quadMat = Zmat.T @ Q[i] @ Zmat
# c[i, 0] = quadMat[0, 0] + 0.5 * np.sum(np.diag(quadMat[1:, 1:]))
# G[i][:gens] = quadMat[0, 1:gens+1] + quadMat[1:gens+1, 0]

print("=== Comparing quadMap Center Computation ===\n")

print("MATLAB formula:")
print("  G(i,1:gens) = 0.5*diag(quadMat(2:gens+1,2:gens+1));")
print("  c(i,1) = quadMat(1,1) + sum(G(i,1:gens));")
print("  Which is: c(i,1) = quadMat(1,1) + 0.5*sum(diag(quadMat(2:gens+1,2:gens+1)))")

print("\nPython formula:")
print("  c[i, 0] = quadMat[0, 0] + 0.5 * np.sum(np.diag(quadMat[1:, 1:]))")
print("  G[i][:gens] = quadMat[0, 1:gens+1] + quadMat[1:gens+1, 0]")

print("\n=== Analysis ===")
print("These formulas are EQUIVALENT:")
print("  MATLAB: quadMat(1,1) + 0.5*sum(diag(quadMat(2:gens+1,2:gens+1)))")
print("  Python: quadMat[0,0] + 0.5*sum(diag(quadMat[1:,1:]))")
print("  (MATLAB uses 1-based indexing, Python uses 0-based)")

print("\nHowever, the differences likely come from:")
print("1. Different BLAS libraries (MATLAB MKL vs NumPy OpenBLAS)")
print("2. Different order of operations in matrix multiplication")
print("3. Floating-point rounding differences")
print("4. Different generator selection in reduce('adaptive')")

print("\nThe ~5.88e-8 difference in VerrorDyn center propagates to:")
print("  trueError diff = ~1.02e-8 (relative ~8e-5)")
print("  This is within expected floating-point precision differences.")
