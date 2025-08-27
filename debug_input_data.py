import numpy as np
import sys
import os

# Add the cora_python path to sys.path
sys.path.insert(0, os.path.abspath('.'))

try:
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    from cora_python.nn.nnHelper import validateNNoptions
    
    print("✓ Successfully imported CORA Python modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

def debug_input_data():
    """Debug the input data loading to understand the dimension mismatch"""
    
    # Paths to the model and specification files
    modelPath = "cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx"
    specPath = "cora_python/examples/nn/models/prop_1.vnnlib"
    
    print(f"DEBUG: Model path: {modelPath}")
    print(f"DEBUG: Spec path: {specPath}")
    
    # Check if files exist
    if not os.path.exists(modelPath):
        print(f"✗ Model file not found: {modelPath}")
        return
    if not os.path.exists(specPath):
        print(f"✗ Specification file not found: {specPath}")
        return
    
    print("✓ Files found")
    
    try:
        # Load the model
        print("\n1. Loading neural network model...")
        nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BC')  # Try 'BC' format instead of 'BSSC'
        print(f"✓ Model loaded successfully")
        print(f"   Model type: {type(nn)}")
        print(f"   Number of layers: {len(nn.layers)}")
        
        # Print layer information
        for i, layer in enumerate(nn.layers):
            print(f"   Layer {i}: {type(layer).__name__}")
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                print(f"     W shape: {layer.W.shape}, b shape: {layer.b.shape}")
        
        # Load specification
        print("\n2. Loading VNNLIB specification...")
        X0, specs = vnnlib2cora(specPath)
        print(f"✓ Specification loaded successfully")
        print(f"   X0 type: {type(X0)}")
        print(f"   X0 length: {len(X0)}")
        
        # Check the first input set
        if X0 and len(X0) > 0:
            first_set = X0[0]
            print(f"   First set type: {type(first_set)}")
            print(f"   First set sup type: {type(first_set.sup)}")
            print(f"   First set inf type: {type(first_set.inf)}")
            
            if hasattr(first_set.sup, 'shape'):
                print(f"   First set sup shape: {first_set.sup.shape}")
            if hasattr(first_set.inf, 'shape'):
                print(f"   First set inf shape: {first_set.inf.shape}")
            
            # Compute center and radius
            print("\n3. Computing center and radius...")
            x = 1/2 * (first_set.sup + first_set.inf)
            r = 1/2 * (first_set.sup - first_set.inf)
            
            print(f"   x type: {type(x)}, shape: {x.shape if hasattr(x, 'shape') else 'no shape'}")
            print(f"   r type: {type(r)}, shape: {r.shape if hasattr(r, 'shape') else 'no shape'}")
            
            # Ensure proper shapes - try different formats
            print("\n4. Testing different input formats:")
            
            # Format 1: Original (5, 1)
            x1 = x.reshape(-1, 1) if hasattr(x, 'ndim') and x.ndim == 1 else x
            r1 = r.reshape(-1, 1) if hasattr(r, 'ndim') and r.ndim == 1 else r
            print(f"   Format 1 (features, batch): x: {x1.shape}, r: {r1.shape}")
            
            # Format 2: Transpose to (1, 5)
            x2 = x1.T
            r2 = r1.T
            print(f"   Format 2 (batch, features): x: {x2.shape}, r: {r2.shape}")
            
            # Format 3: Reshape to (1, 1, 1, 5) for BSSC format
            x3 = x1.reshape(1, 1, 1, -1)
            r3 = r1.reshape(1, 1, 1, -1)
            print(f"   Format 3 (batch, height, width, channels): x: {x3.shape}, r: {r3.shape}")
            
            # Check if dimensions match the first layer
            if nn.layers and hasattr(nn.layers[0], 'W'):
                first_layer_W = nn.layers[0].W
                print(f"\n   First layer W shape: {first_layer_W.shape}")
                
                # Test each format
                for i, (test_x, test_r, format_name) in enumerate([(x1, r1, "Format 1"), (x2, r2, "Format 2"), (x3, r3, "Format 3")], 1):
                    if first_layer_W.shape[1] == test_x.shape[0]:
                        print(f"   ✓ {format_name} matches! {first_layer_W.shape[1]} input features = {test_x.shape[0]} features")
                    else:
                        print(f"   ✗ {format_name} mismatch: {first_layer_W.shape[1]} input features ≠ {test_x.shape[0]} features")
            else:
                print(f"\n   No weight matrix found in first layer")
        
    except Exception as e:
        print(f"✗ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_input_data()
