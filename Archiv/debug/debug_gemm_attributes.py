"""
Check Gemm attributes in ONNX model
"""
import numpy as np
import os
import onnx
import onnxruntime as ort
import sys
sys.path.insert(0, 'cora_python')

from cora_python.g.macros.CORAROOT import CORAROOT

model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'nn-nav-set.onnx')
model = onnx.load(model_path)

print("=== Gemm Node Attributes ===")
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

for i, node in enumerate(model.graph.node):
    if node.op_type == 'Gemm':
        print(f"\nGemm Node {i}: {node.name}")
        print(f"  Inputs: {node.input}")
        
        # Check attributes (defaults per ONNX spec)
        transA = 0
        transB = 1  # Default is 1!
        alpha = 1.0
        beta = 1.0
        
        for attr in node.attribute:
            val = onnx.helper.get_attribute_value(attr)
            if attr.name == 'transA':
                transA = int(val)
            elif attr.name == 'transB':
                transB = int(val)
            elif attr.name == 'alpha':
                alpha = float(val)
            elif attr.name == 'beta':
                beta = float(val)
            print(f"  {attr.name}: {val}")
        
        print(f"\n  Effective attributes:")
        print(f"    transA: {transA} (default 0)")
        print(f"    transB: {transB} (default 1)")
        print(f"    alpha: {alpha} (default 1.0)")
        print(f"    beta: {beta} (default 1.0)")
        
        # Get weight matrix
        if len(node.input) >= 2:
            weight_name = node.input[1]
            if weight_name in initializers:
                W_onnx = initializers[weight_name]
                print(f"\n  Weight '{weight_name}':")
                print(f"    Shape: {W_onnx.shape}")
                print(f"    First row: {W_onnx[0, :] if W_onnx.ndim == 2 else W_onnx[:5]}")
                
                # Gemm computes: Y = alpha * (A^transA) * (B^transB) + beta * C
                # If transB=1 (default), then we use B^T
                # So if B is [out, in], then B^T is [in, out]
                # And we compute: A * B^T
                if transB:
                    W_effective = W_onnx.T
                    print(f"    With transB=1, effective weight shape: {W_effective.shape}")
                else:
                    W_effective = W_onnx
                    print(f"    With transB=0, effective weight shape: {W_effective.shape}")
        
        # Get bias
        if len(node.input) >= 3:
            bias_name = node.input[2]
            if bias_name in initializers:
                b_onnx = initializers[bias_name]
                print(f"  Bias '{bias_name}':")
                print(f"    Shape: {b_onnx.shape}")
                print(f"    Values: {b_onnx.flatten()[:5]}")

# Test with ONNX Runtime
print(f"\n=== ONNX Runtime Test ===")
sess = ort.InferenceSession(model_path)
x = np.ones((1, 4), dtype=np.float32)
y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]
print(f"Input: {x}")
print(f"Output: {y_onnx}")

