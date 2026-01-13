"""Debug script to reproduce the broadcasting error"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.linearSys.linearSys import LinearSys
from cora_python.contDynamics.linearSys.private.priv_correctionMatrixInput import priv_correctionMatrixInput

# Create a simple system
A = np.array([[-1, 0], [0, -2]])
B = np.array([[1], [1]])
C = np.array([[1, 0]])
sys = LinearSys('test_sys', A, B, None, C)

# Create reduced system (simplified)
H_uT = np.array([[1, 2], [3, 4]])
V_uT_proj = np.array([[0.5, 0.3], [0.2, 0.1]])
from cora_python.g.classes.taylorLinSys import TaylorLinSys
linRedSys = LinearSys('reduced_sys_uTrans', H_uT, V_uT_proj.T)
linRedSys.taylor = TaylorLinSys(H_uT)

# Compute G
G = priv_correctionMatrixInput(linRedSys, 0.1, 10)
print(f"G type: {type(G)}")
print(f"G shape: {G.shape}")
print(f"G.inf shape: {G.inf.shape}")
print(f"G.sup shape: {G.sup.shape}")

# Try indexing
G_col = G[:, 0:1]
print(f"\nG[:, 0:1] type: {type(G_col)}")
print(f"G[:, 0:1] shape: {G_col.shape}")
print(f"G[:, 0:1].inf shape: {G_col.inf.shape}")
print(f"G[:, 0:1].sup shape: {G_col.sup.shape}")

# Create V_uT
V_uT = np.random.randn(2, 4)
state_dim = 2
print(f"\nV_uT[:state_dim, :] shape: {V_uT[:state_dim, :].shape}")

# Try the multiplication
try:
    result = V_uT[:state_dim, :] @ G_col
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
