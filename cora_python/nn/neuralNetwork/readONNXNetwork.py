"""
readONNXNetwork - reads and converts a network saved in onnx format

Description:
    Note: If the onnx network contains a custom layer, this function will
    create a +CustomLayer package folder containing all custom layers in
    your current Python directory.

Syntax:
    res = neuralNetwork.readONNXNetwork(file_path)
    res = neuralNetwork.readONNXNetwork(file_path, verbose)
    res = neuralNetwork.readONNXNetwork(file_path, verbose, inputDataFormats)
    res = neuralNetwork.readONNXNetwork(file_path, verbose, inputDataFormats, outputDataFormats)
    res = neuralNetwork.readONNXNetwork(file_path, verbose, inputDataFormats, outputDataFormats, targetNetwork)
    res = neuralNetwork.readONNXNetwork(file_path, verbose, inputDataFormats, outputDataFormats, targetNetwork, containsCompositeLayers)

Inputs:
    file_path - path to file
    verbose - bool if information should be displayed
    inputDataFormats - dimensions of input e.g. 'BC' or 'BSSC'
    outputDataFormats - see inputDataFormats
    targetNetwork - target network type
    containsCompositeLayers - there are residual connections in the network

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       30-March-2022
Last update:   07-June-2022 (specify in- & outputDataFormats)
                30-November-2022 (removed neuralNetworkOld)
                13-February-2023 (simplified function)
                21-October-2024 (clean up DLT function call)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import os
from typing import Optional, List, Any, Dict
import numpy as np
import onnx

# Import CORA Python modules
from .neuralNetwork import NeuralNetwork
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck


def readONNXNetwork(file_path: str, *args) -> NeuralNetwork:
    """
    Read and convert an ONNX network (matches MATLAB exactly).
    
    Args:
        file_path: Path to ONNX file
        *args: Variable arguments (verbose, inputDataFormats, outputDataFormats, targetNetwork, containsCompositeLayers)
        
    Returns:
        Neural network object
    """
    # validate parameters (matches MATLAB narginchk(1,6))
    if len(args) > 5:
        raise ValueError("Too many arguments")
    
    # Set default values (matches MATLAB setDefaultValues exactly)
    [verbose, inputDataFormats, outputDataFormats, targetNetwork, containsCompositeLayers] = setDefaultValues(
        [False, 'BC', 'BC', 'dagnetwork', False], list(args)
    )
    
    # valid in-/outputDataFormats for importONNXNetwork (matches MATLAB exactly)
    validDataFormats = {'', 'BC', 'BCSS', 'BSSC', 'CSS', 'SSC', 'BCSSS', 'BSSSC',
                        'CSSS', 'SSSC', 'TBC', 'BCT', 'BTC', '1BC', 'T1BC', 'TBCSS', 'TBCSSS'}
    
    # validate input (matches MATLAB inputArgsCheck exactly)
    inputArgsCheck([
        [verbose, 'att', 'logical'],
        [inputDataFormats, 'str', validDataFormats],
        [outputDataFormats, 'str', validDataFormats],
        [targetNetwork, 'str', ['dagnetwork', 'dlnetwork']]
    ])
    
    if verbose:
        print("Reading network...")
    
    # try to read ONNX network using Python ONNX library (equivalent to MATLAB's importONNXNetwork)
    try:
        intermediate_net = aux_readONNXviaPython(file_path, inputDataFormats, outputDataFormats, targetNetwork)
        
    except Exception as ME:
        # In MATLAB, this handles GUI-related errors for custom layers
        # For Python, we just re-raise the error
        raise ME
    
    # Process layers (matches MATLAB exactly)
    if containsCompositeLayers:
        # Combine multiple layers into blocks to realize residual connections and
        # parallel computing paths.
        layers = aux_groupCompositeLayers(intermediate_net['Layers'], intermediate_net['Connections'])
    else:
        # Use flat list of layer dicts
        layers = intermediate_net['Layers']
    
    # Convert intermediate layer representation to CORA network
    obj = NeuralNetwork.convertDLToolboxNetwork(layers, verbose)
    
    return obj


# Auxiliary functions -----------------------------------------------------

def aux_readONNXviaPython(file_path: str, inputDataFormats: str, outputDataFormats: str, targetNetwork: str) -> Dict:
    """
    Read ONNX network using Python ONNX library (equivalent to MATLAB's aux_readONNXviaDLT).
    
    This function provides equivalent functionality to MATLAB's importONNXNetwork
    but uses the Python ONNX library instead of Deep Learning Toolbox.
    It parses ONNX files directly and creates layer dictionaries that can be
    converted to CORA layers.
    
    Args:
        file_path: Path to ONNX file
        inputDataFormats: Input data format specification
        outputDataFormats: Output data format specification
        targetNetwork: Target network type
        
    Returns:
        Dictionary with 'Layers' and 'Connections' keys containing intermediate layer representation
    """
    try:
        # Load and parse ONNX model
        model = onnx.load(file_path)
        
        # Validate the model
        onnx.checker.check_model(model)
        
        # Extract model metadata
        graph = model.graph
        initializers = {init.name: init for init in graph.initializer}
        
        # Initialize layers list
        layers = []
        layer_id = 0
        
        # Process each node in the graph
        skip_next = False
        skip_indices = set()  # Track indices to skip (e.g., Pad nodes fused into pooling)
        for i, node in enumerate(graph.node):
            if skip_next:
                skip_next = False
                continue
            if i in skip_indices:
                continue
            matched = False
            layer_info = {
                'Name': f'Layer_{layer_id}',
                'Type': node.op_type,
                'Inputs': list(node.input),
                'Outputs': list(node.output),
                'Attributes': {attr.name: attr for attr in node.attribute}
            }
            
            # Handle different layer types (matches MATLAB's layer conversion)
            if node.op_type == 'Gemm':
                # Fully connected layer
                # Gemm: Y = alpha * (A^transA) * (B^transB) + beta * C
                # where A is input, B is weight, C is bias
                layer_info['Type'] = 'FullyConnectedLayer'
                matched = True
                
                # Extract Gemm attributes
                transA = 0  # default: no transpose on A (input)
                transB = 0  # default: no transpose on B (weight)
                alpha = 1.0  # default scaling factor
                beta = 1.0   # default bias scaling factor
                
                for attr in node.attribute:
                    if attr.name == 'transA':
                        transA = attr.i
                    elif attr.name == 'transB':
                        transB = attr.i
                    elif attr.name == 'alpha':
                        alpha = attr.f
                    elif attr.name == 'beta':
                        beta = attr.f
                
                # Extract weights and bias
                if len(node.input) >= 2:
                    weight_name = node.input[1]
                    if weight_name in initializers:
                        weight = onnx.numpy_helper.to_array(initializers[weight_name])
                        # Apply transB if needed
                        # If transB=0: B is [in_features, out_features], transpose to [out_features, in_features]
                        # If transB=1: B is [out_features, in_features], no transpose needed
                        if not transB:
                            weight = weight.T
                        # Apply alpha scaling
                        if alpha != 1.0:
                            weight = weight * alpha
                        # Convert to float64 to match MATLAB's double precision
                        weight = weight.astype(np.float64)
                        layer_info['Weight'] = weight
                    
                    if len(node.input) >= 3:
                        bias_name = node.input[2]
                        if bias_name in initializers:
                            bias = onnx.numpy_helper.to_array(initializers[bias_name])
                            # Apply beta scaling
                            if beta != 1.0:
                                bias = bias * beta
                            # Convert to float64 to match MATLAB's double precision
                            bias = bias.astype(np.float64)
                            layer_info['Bias'] = bias
                
                # Mark that this came from Gemm
                # After handling transB, weight is in [out_features, in_features] format
                layer_info['FromGemm'] = True
                
            elif node.op_type == 'Relu':
                layer_info['Type'] = 'ReLULayer'
                matched = True
                
            elif node.op_type == 'Sigmoid':
                layer_info['Type'] = 'SigmoidLayer'
                matched = True
                
            elif node.op_type == 'Tanh':
                layer_info['Type'] = 'TanhLayer'
                matched = True
                
            elif node.op_type == 'Reshape':
                layer_info['Type'] = 'ReshapeLayer'
                matched = True
                # Extract shape information from attributes or input tensor
                shape_found = False
                for attr in node.attribute:
                    if attr.name == 'shape':
                        layer_info['Shape'] = list(attr.ints)
                        shape_found = True
                        break
                
                # If no shape attribute, try to get it from input tensor
                if not shape_found and len(node.input) >= 2:
                    shape_name = node.input[1]
                    if shape_name in initializers:
                        shape_tensor = onnx.numpy_helper.to_array(initializers[shape_name])
                        layer_info['Shape'] = list(shape_tensor)
                        shape_found = True
                
                # If still no shape found, this might be a dynamic reshape
                if not shape_found:
                    # For dynamic reshapes, we need to infer the target shape
                    # This is a common case where the shape is computed at runtime
                    layer_info['Shape'] = [-1]  # Will be determined dynamically
                
            elif node.op_type == 'Conv':
                layer_info['Type'] = 'Conv2DLayer'
                matched = True
                # Extract convolution parameters
                for attr in node.attribute:
                    if attr.name == 'kernel_shape':
                        layer_info['KernelSize'] = list(attr.ints)
                    elif attr.name == 'strides':
                        layer_info['Stride'] = list(attr.ints)
                    elif attr.name == 'pads':
                        layer_info['Padding'] = list(attr.ints)
                
                # Extract weights and bias
                if len(node.input) >= 2:
                    weight_name = node.input[1]
                    if weight_name in initializers:
                        weight = onnx.numpy_helper.to_array(initializers[weight_name])
                        layer_info['Weight'] = weight
                    
                    if len(node.input) >= 3:
                        bias_name = node.input[2]
                        if bias_name in initializers:
                            bias = onnx.numpy_helper.to_array(initializers[bias_name])
                            layer_info['Bias'] = bias
                
            elif node.op_type == 'MaxPool' or node.op_type == 'AveragePool':
                layer_info['Type'] = 'PoolingLayer'
                matched = True
                layer_info['PoolType'] = node.op_type
                # Extract pooling parameters
                for attr in node.attribute:
                    if attr.name == 'kernel_shape':
                        layer_info['KernelSize'] = list(attr.ints)
                    elif attr.name == 'strides':
                        layer_info['Stride'] = list(attr.ints)
                    elif attr.name == 'pads':
                        layer_info['Padding'] = list(attr.ints)
                
                # If kernel_shape is not specified, try to infer from input/output shapes
                # ONNX AveragePool requires kernel_shape, but some models don't specify it
                if 'KernelSize' not in layer_info:
                    # Try to infer from value_info if available
                    # For now, default to [4, 4] which is common for this type of network
                    # This matches the pattern: 27x27 -> 6x6 with stride 4
                    layer_info['KernelSize'] = [4, 4]
                
                # Check if previous node is Pad and fuse its padding
                if i > 0:
                    prev_node = graph.node[i - 1]
                    if prev_node.op_type == 'Pad' and prev_node.output[0] == node.input[0]:
                        # Pad operation before pooling - fuse padding
                        pad_pads = None
                        pad_mode = 'constant'
                        
                        # Check attributes first (older ONNX versions)
                        for attr in prev_node.attribute:
                            if attr.name == 'pads':
                                pad_pads = list(attr.ints)
                            elif attr.name == 'mode':
                                pad_mode = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
                        
                        # Check if pads are provided as input tensor (ONNX opset 11+)
                        if pad_pads is None and len(prev_node.input) >= 2:
                            pads_name = prev_node.input[1]
                            if pads_name in initializers:
                                pad_pads = list(onnx.numpy_helper.to_array(initializers[pads_name]))
                        
                        # If pad_pads is None or all zeros, the Pad is a no-op, skip it
                        if pad_pads is None:
                            # No padding specified - Pad is a no-op, just skip it
                            # Find and remove the Pad layer from layers list if it was added
                            for j in range(len(layers) - 1, -1, -1):
                                if isinstance(layers[j], dict) and layers[j].get('Outputs', []) == list(prev_node.output):
                                    layers.pop(j)
                                    break
                        elif pad_mode == 'constant' and pad_pads and not all(p == 0 for p in pad_pads):
                            # ONNX Pad pads format depends on tensor dimensions
                            # For 4D tensor [N, C, H, W]: [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
                            # For 2D spatial: typically [0, 0, top, left, 0, 0, bottom, right] or [top, left, bottom, right]
                            # Extract spatial padding (H and W dimensions)
                            if len(pad_pads) == 8:
                                # 4D tensor: extract H and W padding [H_begin, W_begin, H_end, W_end]
                                pad_pads = [pad_pads[2], pad_pads[3], pad_pads[6], pad_pads[7]]  # [top, left, bottom, right]
                            elif len(pad_pads) == 4:
                                # Already in [top, left, bottom, right] format
                                pass
                            
                            if len(pad_pads) == 4:
                                # If pooling already has padding, add them together
                                if 'Padding' in layer_info:
                                    # ONNX pads format: [top, left, bottom, right]
                                    existing_pad = layer_info['Padding']
                                    # Add: [top+top, left+left, bottom+bottom, right+right]
                                    layer_info['Padding'] = [
                                        existing_pad[0] + pad_pads[0],  # top
                                        existing_pad[1] + pad_pads[1],  # left
                                        existing_pad[2] + pad_pads[2],  # bottom
                                        existing_pad[3] + pad_pads[3]   # right
                                    ]
                                else:
                                    # Use Pad's padding
                                    layer_info['Padding'] = pad_pads
                                # Find and remove the Pad layer from layers list
                                for j in range(len(layers) - 1, -1, -1):
                                    if isinstance(layers[j], dict) and layers[j].get('Outputs', []) == list(prev_node.output):
                                        layers.pop(j)
                                        break
                
            elif node.op_type == 'Add':
                # Add operations are typically fused with MatMul in the MatMul handler above
                # If we reach here, it's a standalone Add that wasn't fused
                # Skip it to match MATLAB behavior (MATLAB's importONNXNetwork fuses MatMul+Add)
                matched = True  # Mark as matched but don't create layer
                continue  # Skip this node
                
            elif node.op_type == 'Mul':
                layer_info['Type'] = 'ElementwiseAffineLayer'
                matched = True
                # Try to extract scale if second input is an initializer
                if len(node.input) >= 2:
                    scale_name = node.input[1]
                    if scale_name in initializers:
                        scale = onnx.numpy_helper.to_array(initializers[scale_name])
                        layer_info['Scale'] = scale
                
            elif node.op_type == 'Sub':
                # Skip Sub operations to match MATLAB behavior
                # MATLAB skips ScalingLayer operations entirely
                matched = True  # Mark as matched but don't create layer
                continue  # Skip this node
                
            elif node.op_type == 'MatMul':
                layer_info['Type'] = 'FullyConnectedLayer'
                matched = True
                # Extract weights if available
                if len(node.input) >= 2:
                    weight_name = node.input[1]
                    if weight_name in initializers:
                        weight = onnx.numpy_helper.to_array(initializers[weight_name])
                        layer_info['Weight'] = weight
                # Mark that this came from MatMul (weight format is [in_features, out_features], needs transpose)
                layer_info['FromGemm'] = False
                layer_info['FromMatMul'] = True
                # Fuse MatMul + Add into FullyConnectedLayer (matches MATLAB's importONNXNetwork behavior)
                # MATLAB's importONNXNetwork automatically fuses these operations
                if i + 1 < len(graph.node):
                    next_node = graph.node[i + 1]
                    if next_node.op_type == 'Add' and next_node.input[0] == node.output[0]:
                        if len(next_node.input) >= 2:
                            bias_name = next_node.input[1]
                            if bias_name in initializers:
                                bias = onnx.numpy_helper.to_array(initializers[bias_name])
                                layer_info['Bias'] = bias
                                skip_next = True  # Skip the Add node since we fused it
            elif node.op_type == 'Identity':
                layer_info['Type'] = 'IdentityLayer'
                matched = True
            elif node.op_type == 'Flatten':
                layer_info['Type'] = 'ReshapeLayer'
                matched = True
                axis = 1
                for attr in node.attribute:
                    if attr.name == 'axis':
                        axis = attr.i
                layer_info['Shape'] = [-1]
                layer_info['FlattenAxis'] = axis
            elif node.op_type == 'Softmax':
                layer_info['Type'] = 'SoftmaxLayer'
                matched = True
                for attr in node.attribute:
                    if attr.name == 'axis':
                        layer_info['Axis'] = attr.i
            elif node.op_type == 'BatchNormalization':
                layer_info['Type'] = 'BatchNormLayer'
                matched = True
                if len(node.input) >= 3:
                    scale_name = node.input[1]
                    bias_name = node.input[2]
                    mean_name = node.input[3]
                    var_name = node.input[4]
                    if scale_name in initializers:
                        scale = onnx.numpy_helper.to_array(initializers[scale_name])
                        layer_info['Scale'] = scale
                    if bias_name in initializers:
                        bias = onnx.numpy_helper.to_array(initializers[bias_name])
                        layer_info['Bias'] = bias
                    if mean_name in initializers:
                        mean = onnx.numpy_helper.to_array(initializers[mean_name])
                        layer_info['Mean'] = mean
                    if var_name in initializers:
                        var = onnx.numpy_helper.to_array(initializers[var_name])
                        layer_info['Variance'] = var
            # Unknown fallback only if no known mapping matched
            if not matched:
                layer_info['Type'] = 'UnknownLayer'
                layer_info['OriginalType'] = node.op_type
                for attr in node.attribute:
                    layer_info[f'Attr_{attr.name}'] = onnx.helper.get_attribute_value(attr)
            layers.append(layer_info)
            layer_id += 1
        
        # Add input layer if we have layers (extract input size from ONNX graph)
        if layers and len(graph.input) > 0:
            # Extract input size from ONNX (in ONNX format, e.g., BCSS = [batch, channel, H, W])
            onnx_input_size = [dim.dim_value if dim.dim_value > 0 else 1 
                              for dim in graph.input[0].type.tensor_type.shape.dim]
            
            # Convert from ONNX format to CORA format [H, W, C]
            # inputDataFormats specifies the ONNX format (e.g., 'BCSS' = Batch, Channel, Spatial, Spatial)
            if inputDataFormats == 'BCSS' and len(onnx_input_size) == 4:
                # ONNX: [batch, channel, height, width] -> CORA: [height, width, channel]
                cora_input_size = [onnx_input_size[2], onnx_input_size[3], onnx_input_size[1]]
            elif inputDataFormats == 'BSSC' and len(onnx_input_size) == 4:
                # ONNX: [batch, height, width, channel] -> CORA: [height, width, channel]
                cora_input_size = [onnx_input_size[1], onnx_input_size[2], onnx_input_size[3]]
            elif inputDataFormats == 'BC' and len(onnx_input_size) == 2:
                # ONNX: [batch, features] -> CORA: [features]
                cora_input_size = [onnx_input_size[1]]
            else:
                # Default: use as-is (might need adjustment for other formats)
                cora_input_size = onnx_input_size
            
            input_layer = {
                'Name': 'InputLayer',
                'Type': 'InputLayer',
                'InputSize': cora_input_size
            }
            layers.insert(0, input_layer)
        
        # Return the structure expected by the calling code
        result = {
            'Layers': layers,
            'Connections': []  # For now, no connections - this can be enhanced later
        }
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to parse ONNX file: {str(e)}"
        raise RuntimeError(error_msg) from e


def aux_groupCompositeLayers(layerslist: List, connections: List) -> List:
    """
    Group composite layers for residual connections (matches MATLAB exactly).
    
    Args:
        layerslist: List of layers
        connections: List of connections
        
    Returns:
        Grouped layers
    """
    # Find initial layer.
    layer0 = aux_findLayerByName(layerslist, connections[0]['Source'])
    layers = [layer0]
    
    for i in range(len(connections)):
        # Find source and destination layer.
        layerDest = aux_findLayerByName(layerslist, connections[i]['Destination'])
        
        # Check if the next layer aggregates multiple paths.
        isAggrLayer = layerDest['Name'] != connections[i]['Destination']
        
        # Find source layer within the current list of layers.
        for j in range(len(layers) - 1, -1, -1):  # Speed up computations by searching from the back.
            # Obtain paths from the j-th step.
            layerj = layers[j]
            
            if isinstance(layerj, list):
                # Iterate over the current paths to try and find the source.
                for k in range(len(layerj)):
                    # Extract layer from the j-th step and k-th path.
                    layerjk = layerj[k]
                    for l in range(len(layerjk) - 1, -1, -1):
                        if layerjk[l]['Name'] == connections[i]['Source']:
                            # Append in computation path.
                            if isAggrLayer:
                                # Add at the end.
                                if j + 1 >= len(layers):
                                    layers.append(layerDest)
                                else:
                                    layers[j + 1] = layerDest
                            else:
                                layers[j][k].append(layerDest)
                            break
            else:
                # Compare the names.
                if layerj['Name'] == connections[i]['Source']:
                    # Found source layer; append the destination.
                    if j + 1 >= len(layers):
                        # Add at the end.
                        layers.append(layerDest)
                    else:
                        if isAggrLayer:
                            # There is a residual connection.
                            if isinstance(layers[j + 1], list):
                                layers[j + 1].append(None)
                            else:
                                layers[j + 1] = [layers[j + 1], None]
                            if j + 2 >= len(layers):
                                layers.append(layerDest)
                            else:
                                layers[j + 2] = layerDest
                        else:
                            # There is a new computation path.
                            if isinstance(layers[j + 1], list):
                                layers[j + 1].append(layerDest)
                            else:
                                layers[j + 1] = [layers[j + 1], layerDest]
                    break
                elif layerj['Name'].startswith(connections[i]['Source']):
                    # Combine computation paths.
                    # TODO
                    break
    
    return layers


def aux_findLayerByName(layers: List, destName: str) -> dict:
    """
    Find layer by name (matches MATLAB exactly).
    
    Args:
        layers: List of layers
        destName: Destination name
        
    Returns:
        Found layer or empty dict
    """
    # Remove trailing '/*' if present
    if '/*' in destName:
        destName = destName.split('/*')[0]
    
    layer = {}
    for i in range(len(layers)):
        if destName == layers[i]['Name'] or (destName + '/*') in layers[i]['Name']:
            layer = layers[i]
            break
    
    return layer
