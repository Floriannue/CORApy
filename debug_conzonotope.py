import numpy as np
from cora_python.contSet.conZonotope.conZonotope import ConZonotope

# Test the ConZonotope constructor that's failing in the test (with float dtypes)
c = np.array([[0.0], [0.0]])
G = np.array([[1.5, -1.5, 0.5], [1.0, 0.5, -1.0]])
A = np.array([[1.0, 1.0, 1.0]])
b = np.array([1.0])

print("c shape:", c.shape, "dtype:", c.dtype)
print("c:\n", c)
print("G shape:", G.shape, "dtype:", G.dtype)
print("G:\n", G)
print("A shape:", A.shape, "dtype:", A.dtype)
print("A:\n", A)
print("b shape:", b.shape, "dtype:", b.dtype)
print("b:\n", b)

# Check if arrays contain finite values
print("c finite:", np.all(np.isfinite(c)))
print("G finite:", np.all(np.isfinite(G)))
print("A finite:", np.all(np.isfinite(A)))
print("b finite:", np.all(np.isfinite(b)))

# Check the combined array [c, G] which is the first input
combined = np.hstack([c, G])
print("Combined [c, G] shape:", combined.shape, "dtype:", combined.dtype)
print("Combined finite:", np.all(np.isfinite(combined)))

try:
    cZ = ConZonotope(c, G, A, b)
    print("ConZonotope created successfully")
except Exception as e:
    print("Error:", e)
    
# Try with explicit conversion
try:
    print("\nTrying with explicit float conversion...")
    c_f = c.astype(np.float64)
    G_f = G.astype(np.float64) 
    A_f = A.astype(np.float64)
    b_f = b.astype(np.float64)
    cZ = ConZonotope(c_f, G_f, A_f, b_f)
    print("ConZonotope created successfully with explicit conversion")
except Exception as e:
    print("Error with explicit conversion:", e) 