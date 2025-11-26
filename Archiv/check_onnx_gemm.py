import onnx
import numpy as np
import sys
sys.path.insert(0, 'cora_python')
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.nn.neuralNetwork import NeuralNetwork

model_path = f"{CORAROOT()}/cora_matlab/models/Cora/nn/nn-nav-set.onnx"
model = onnx.load(model_path)
initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

# Find first Gemm
nodes = [n for n in model.graph.node if n.op_type == 'Gemm']
if nodes:
    n = nodes[0]
    print(f"First Gemm node: {n.name}")
    print(f"  Inputs: {list(n.input)}")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
    print(f"  Attributes: {attrs}")
    
    w_name = n.input[1] if len(n.input) > 1 else None
    if w_name and w_name in initializers:
        w = initializers[w_name]
        print(f"  Weight '{w_name}' shape: {w.shape}")
        print(f"  Weight first row: {w[0, :5] if w.shape[1] >= 5 else w[0, :]}")

# Check CORA first layer
nn = NeuralNetwork.readONNXNetwork(model_path)
if len(nn.layers) > 0 and hasattr(nn.layers[0], 'W'):
    w_cora = nn.layers[0].W
    print(f"\nCORA first layer W shape: {w_cora.shape}")
    print(f"CORA first row: {w_cora[0, :5] if w_cora.shape[1] >= 5 else w_cora[0, :]}")
    
    if w_name and w_name in initializers:
        w_onnx = initializers[w_name]
        print(f"\nComparison:")
        print(f"  ONNX W shape: {w_onnx.shape}")
        print(f"  CORA W shape: {w_cora.shape}")
        print(f"  W_onnx == W_cora? {np.allclose(w_onnx, w_cora)}")
        print(f"  W_onnx == W_cora.T? {np.allclose(w_onnx, w_cora.T)}")
        print(f"  W_onnx.T == W_cora? {np.allclose(w_onnx.T, w_cora)}")

