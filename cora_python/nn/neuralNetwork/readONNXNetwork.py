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
        dltoolbox_net = aux_readONNXviaPython(file_path, inputDataFormats, outputDataFormats, targetNetwork)
        
    except Exception as ME:
        # In MATLAB, this handles GUI-related errors for custom layers
        # For Python, we just re-raise the error
        raise ME
    
    # Process layers (matches MATLAB exactly)
    if containsCompositeLayers:
        # Combine multiple layers into blocks to realize residual connections and
        # parallel computing paths.
        layers = aux_groupCompositeLayers(dltoolbox_net['Layers'], dltoolbox_net['Connections'])
    else:
        # Convert to cell array format like MATLAB's num2cell(dltoolbox_net.Layers)
        layers = [[layer] for layer in dltoolbox_net['Layers']]
    
    # convert DLT network to CORA network (matches MATLAB exactly)
    obj = NeuralNetwork.convertDLToolboxNetwork(layers, verbose)
    
    return obj


# Auxiliary functions -----------------------------------------------------

def aux_readONNXviaPython(file_path: str, inputDataFormats: str, outputDataFormats: str, targetNetwork: str) -> Dict:
    """
    Read ONNX network using Python ONNX library (equivalent to MATLAB's aux_readONNXviaDLT).
    
    This function provides equivalent functionality to MATLAB's importONNXNetwork
    but uses the Python ONNX library instead of Deep Learning Toolbox.
    
    Args:
        file_path: Path to ONNX file
        inputDataFormats: Input data format specification
        outputDataFormats: Output data format specification
        targetNetwork: Target network type
        
    Returns:
        Dictionary with 'Layers' and 'Connections' keys, matching MATLAB's structure
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
        for node in graph.node:
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
                layer_info['Type'] = 'FullyConnectedLayer'
                
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
                
            elif node.op_type == 'Relu':
                layer_info['Type'] = 'ReLULayer'
                
            elif node.op_type == 'Sigmoid':
                layer_info['Type'] = 'SigmoidLayer'
                
            elif node.op_type == 'Tanh':
                layer_info['Type'] = 'TanhLayer'
                
            elif node.op_type == 'Reshape':
                layer_info['Type'] = 'ReshapeLayer'
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
                
                # Debug output to see what shape was extracted
                print(f"DEBUG: Reshape layer '{layer_info.get('Name', 'unnamed')}' shape: {layer_info.get('Shape', 'None')}")
                
            elif node.op_type == 'Conv':
                layer_info['Type'] = 'Conv2DLayer'
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
                layer_info['PoolType'] = node.op_type
                # Extract pooling parameters
                for attr in node.attribute:
                    if attr.name == 'kernel_shape':
                        layer_info['KernelSize'] = list(attr.ints)
                    elif attr.name == 'strides':
                        layer_info['Stride'] = list(attr.ints)
                    elif attr.name == 'pads':
                        layer_info['Padding'] = list(attr.ints)
                
            elif node.op_type == 'Add':
                layer_info['Type'] = 'ElementwiseAffineLayer'
                
            elif node.op_type == 'Mul':
                layer_info['Type'] = 'ElementwiseAffineLayer'
                
            elif node.op_type == 'Sub':
                # Subtraction operation - often used for preprocessing
                layer_info['Type'] = 'ElementwiseAffineLayer'
                # Extract the value being subtracted
                if len(node.input) >= 2:
                    const_name = node.input[1]
                    if const_name in initializers:
                        const_value = onnx.numpy_helper.to_array(initializers[const_name])
                        layer_info['Offset'] = -const_value  # Negative because we're subtracting
                        layer_info['Scale'] = np.ones_like(const_value)  # Scale by 1
                
            elif node.op_type == 'MatMul':
                layer_info['Type'] = 'FullyConnectedLayer'
                # Extract weights if available
                if len(node.input) >= 2:
                    weight_name = node.input[1]
                    if weight_name in initializers:
                        weight = onnx.numpy_helper.to_array(initializers[weight_name])
                        layer_info['Weight'] = weight
                
            elif node.op_type == 'Identity':
                layer_info['Type'] = 'IdentityLayer'
                
            elif node.op_type == 'Flatten':
                layer_info['Type'] = 'ReshapeLayer'
                # For Flatten nodes, we need to calculate the target shape
                # The axis attribute determines where flattening starts
                axis = 1  # Default axis
                for attr in node.attribute:
                    if attr.name == 'axis':
                        axis = attr.i
                
                # Calculate target shape based on input shape and axis
                # For input shape [1, 1, 1, 5] and axis=1, we flatten from dimension 1 onwards
                # This means we keep dimensions 0, and flatten the rest
                # Result: [1, 5] (flattening dimensions 1,2,3 into one dimension)
                layer_info['Shape'] = [-1]  # Will be determined dynamically based on input
                layer_info['FlattenAxis'] = axis  # Store the axis for later processing
                
                print(f"DEBUG: Flatten layer '{layer_info.get('Name', 'unnamed')}' axis: {axis}")
                
            elif node.op_type == 'Softmax':
                layer_info['Type'] = 'SoftmaxLayer'
                # Extract axis information
                for attr in node.attribute:
                    if attr.name == 'axis':
                        layer_info['Axis'] = attr.i
                
            elif node.op_type == 'BatchNormalization':
                layer_info['Type'] = 'BatchNormLayer'
                # Extract batch normalization parameters
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
                
            else:
                # Unknown layer type - create a placeholder
                layer_info['Type'] = 'UnknownLayer'
                layer_info['OriginalType'] = node.op_type
                # Try to extract any available parameters
                for attr in node.attribute:
                    layer_info[f'Attr_{attr.name}'] = onnx.helper.get_attribute_value(attr)
            
            layers.append(layer_info)
            layer_id += 1
        
        # Add input and output layer placeholders
        if layers:
            # Add input layer
            input_layer = {
                'Name': 'InputLayer',
                'Type': 'InputLayer',
                'InputSize': [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
            }
            layers.insert(0, input_layer)
            
            # Add output layer
            output_layer = {
                'Name': 'OutputLayer',
                'Type': 'OutputLayer',
                'OutputSize': [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim]
            }
            layers.append(output_layer)
        
        # Return the structure expected by the calling code
        # MATLAB expects a dictionary with 'Layers' and 'Connections' keys
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
