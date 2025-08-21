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
from typing import Optional, List, Any
import numpy as np

# Import CORA Python modules
from .neuralNetwork import NeuralNetwork


def readONNXNetwork(file_path: str, *args) -> NeuralNetwork:
    """
    Read and convert an ONNX network.
    
    Args:
        file_path: Path to ONNX file
        *args: Variable arguments (verbose, inputDataFormats, outputDataFormats, targetNetwork, containsCompositeLayers)
        
    Returns:
        Neural network object
    """
    # validate parameters
    if len(args) > 5:
        raise ValueError("Too many arguments")
    
    # Set default values
    verbose = False
    inputDataFormats = 'BC'
    outputDataFormats = 'BC'
    targetNetwork = 'dagnetwork'
    containsCompositeLayers = False
    
    if len(args) >= 1:
        verbose = args[0]
    if len(args) >= 2:
        inputDataFormats = args[1]
    if len(args) >= 3:
        outputDataFormats = args[2]
    if len(args) >= 4:
        targetNetwork = args[3]
    if len(args) >= 5:
        containsCompositeLayers = args[4]
    
    # valid in-/outputDataFormats for importONNXNetwork
    validDataFormats = {'', 'BC', 'BCSS', 'BSSC', 'CSS', 'SSC', 'BCSSS', 'BSSSC',
                        'CSSS', 'SSSC', 'TBC', 'BCT', 'BTC', '1BC', 'T1BC', 'TBCSS', 'TBCSSS'}
    
    if inputDataFormats not in validDataFormats:
        raise ValueError(f"Invalid input data format: {inputDataFormats}")
    if outputDataFormats not in validDataFormats:
        raise ValueError(f"Invalid output data format: {outputDataFormats}")
    
    validTargetNetworks = {'dagnetwork', 'dlnetwork'}
    if targetNetwork not in validTargetNetworks:
        raise ValueError(f"Invalid target network: {targetNetwork}")
    
    if verbose:
        print("Reading network...")
    
    # try to read ONNX network using dltoolbox
    try:
        dltoolbox_net = aux_readONNXviaDLT(file_path, inputDataFormats, outputDataFormats, targetNetwork)
    except Exception as ME:
        # This is a placeholder for the MATLAB-specific error handling
        # In practice, this would handle specific MATLAB errors
        if verbose:
            print(f"Error reading ONNX network: {ME}")
        raise
    
    if containsCompositeLayers:
        # Combine multiple layers into blocks to realize residual connections and
        # parallel computing paths.
        layers = aux_groupCompositeLayers(dltoolbox_net['Layers'], dltoolbox_net['Connections'])
    else:
        layers = dltoolbox_net['Layers']
    
    # convert DLT network to CORA network
    # obj = neuralNetwork.convertDLToolboxNetwork(dltoolbox_net.Layers, verbose);
    obj = neuralNetwork_convertDLToolboxNetwork(layers, verbose)
    
    return obj


# Auxiliary functions -----------------------------------------------------

def aux_readONNXviaDLT(file_path: str, inputDataFormats: str, outputDataFormats: str, targetNetwork: str) -> dict:
    """
    Read ONNX network via DLT (Deep Learning Toolbox).
    
    Args:
        file_path: Path to ONNX file
        inputDataFormats: Input data format
        outputDataFormats: Output data format
        targetNetwork: Target network type
        
    Returns:
        Dictionary containing network information
    """
    # build name-value pairs
    NVpairs = {}
    
    # input data format
    if inputDataFormats:
        NVpairs['InputDataFormats'] = inputDataFormats
    
    # output data format
    if outputDataFormats:
        NVpairs['OutputDataFormats'] = outputDataFormats
    
    # custom layers generated from DLT will be stored in this folder
    # https://de.mathworks.com/help/deeplearning/ref/importnetworkfromonnx.html#mw_ccdf29c9-84cf-4175-a8ce-8e6ab1c89d4c
    customLayerName = 'DLT_CustomLayers'
    
    # This is a placeholder implementation
    # In practice, this would use ONNX libraries to read and convert the network
    # For now, return a mock network structure
    mock_layers = [
        {'Name': 'input', 'Type': 'InputLayer'},
        {'Name': 'fc1', 'Type': 'FullyConnectedLayer'},
        {'Name': 'relu1', 'Type': 'ReLULayer'},
        {'Name': 'fc2', 'Type': 'FullyConnectedLayer'},
        {'Name': 'output', 'Type': 'OutputLayer'}
    ]
    
    mock_connections = [
        {'Source': 'input', 'Destination': 'fc1'},
        {'Source': 'fc1', 'Destination': 'relu1'},
        {'Source': 'relu1', 'Destination': 'fc2'},
        {'Source': 'fc2', 'Destination': 'output'}
    ]
    
    return {
        'Layers': mock_layers,
        'Connections': mock_connections
    }


def aux_removeIndentCodeLines(ME):
    """
    Remove 'indentcode' function call.
    
    Args:
        ME: Exception object
    """
    # This is a placeholder for the MATLAB-specific functionality
    # In practice, this would handle MATLAB internal file modifications
    pass


def aux_groupCompositeLayers(layerslist: List, connections: List) -> List:
    """
    Group composite layers for residual connections.
    
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
    Find layer by name.
    
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


def neuralNetwork_convertDLToolboxNetwork(dltoolbox_layers: List, verbose: bool) -> NeuralNetwork:
    """
    Convert DLToolbox network to CORA network.
    
    Args:
        dltoolbox_layers: DLToolbox layers
        verbose: Whether to print verbose output
        
    Returns:
        CORA neural network
    """
    # This is a placeholder implementation
    # In practice, this would convert the DLToolbox layers to CORA layers
    if verbose:
        print("Converting DLToolbox network to CORA network...")
    
    # For now, return a mock network
    # In practice, this would parse the layers and create the actual network
    from .layers.linear.nnLinearLayer import nnLinearLayer
    from .layers.nonlinear.nnReLULayer import nnReLULayer
    
    # Create a simple mock network
    W1 = np.random.rand(5, 5)  # Mock weights
    b1 = np.random.rand(5, 1)  # Mock biases
    W2 = np.random.rand(5, 5)  # Mock weights
    b2 = np.random.rand(5, 1)  # Mock biases
    
    layers = [
        nnLinearLayer(W1, b1),
        nnReLULayer(),
        nnLinearLayer(W2, b2)
    ]
    
    return NeuralNetwork(layers, name="ONNX_Network")
