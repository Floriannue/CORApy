"""
prepareForZonoBatchEval - prepare neural network for zonotope batch
  evaluation: In each layer store the active generator ids and identity 
  matrices to quickly add approximation errors.

Syntax:
    nn.prepareForZonoBatchEval(x, numInitGens, useApproxError, idxLayer)

Inputs:
    x - example input; used for size and type
    options
    idxLayer - indices of layers that should be evaluated

Outputs:
    numGen - number of generators used during training

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/evaluateZonotopeBatch

Authors:       Lukas Koller
Written:       07-February-2024
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .neuralNetwork import NeuralNetwork
from ..nnHelper import validateNNoptions


def prepareForZonoBatchEval(nn: NeuralNetwork, x: np.ndarray, 
                           options: Optional[Dict[str, Any]] = None, 
                           idxLayer: Optional[List[int]] = None) -> int:
    """
    Prepare neural network for zonotope batch evaluation
    
    Args:
        nn: NeuralNetwork object
        x: example input; used for size and type
        options: evaluation options
        idxLayer: indices of layers that should be evaluated (defaults to all layers)
        
    Returns:
        numGen: number of generators used during training
    """
    v0 = x.shape[0]
    
    # validate parameters
    if options is None:
        options = {}
    if idxLayer is None:
        idxLayer = list(range(len(nn.layers)))  # 0-based indexing like Python
    
    options = validateNNoptions(options, True)
    
    # compute total number of generators
    if options.get('nn', {}).get('train', {}).get('num_init_gens', float('inf')) < float('inf'):
        numGen = options['nn']['train']['num_init_gens']
    else:
        numGen = v0
    
    numApproxErr = options.get('nn', {}).get('train', {}).get('num_approx_err', 0)
    numNeurons = v0  # store number of neurons of the current layer
    
    for i in idxLayer:
        # Extract current layer.
        layeri = nn.layers[i]  # Convert to 0-based indexing
        
        # Handle the sublayers of composite layers separately.
        if hasattr(layeri, '__class__') and 'nnCompositeLayer' in str(layeri.__class__):
            for j in range(len(layeri.layers)):
                # Extract the j-th computation path.
                layerij = layeri.layers[j]
                # Iterate over the layers of the current computation path.
                for k in range(len(layerij)):
                    # Prepare the current layer and update the number of 
                    # generators and neurons.
                    numGen, numNeurons = _aux_prepareLayer(layerij[k], 
                                                         numApproxErr, numGen, numNeurons, options)
        else:
            # Prepare the current layer and update the number of generators and
            # neurons.
            numGen, numNeurons = _aux_prepareLayer(layeri, numApproxErr, 
                                                 numGen, numNeurons, options)
    
    return numGen


def _aux_prepareLayer(layeri: Any, numApproxErr: int, numGen: int, 
                     numNeurons: int, options: Dict[str, Any]) -> Tuple[int, int]:
    """
    Auxiliary function to prepare a single layer
    
    Args:
        layeri: layer to prepare
        numApproxErr: number of approximation errors
        numGen: current number of generators
        numNeurons: current number of neurons
        options: evaluation options
        
    Returns:
        Tuple of (numGen, numNeurons) updated values
    """
    # Store number generators of the input
    if not hasattr(layeri, 'backprop'):
        layeri.backprop = {'store': {}}
    # 1-based indexing like MATLAB
    layeri.backprop['store']['genIds'] = list(range(numGen))  # 0-based indexing like Python
    
    if hasattr(layeri, '__class__') and 'nnGeneratorReductionLayer' in str(layeri.__class__):
        layeri.maxGens = min(numGen, layeri.maxGens)
        numGen = layeri.maxGens
    elif hasattr(layeri, '__class__') and 'nnActivationLayer' in str(layeri.__class__):
        if numApproxErr > 0 and not options.get('nn', {}).get('interval_center', False):
            # We store an id-matrix in each activation layer to append 
            # the approximation errors of image enclosures to the 
            # generator matrix. This eliminates the need to allocate 
            # new GPU memory during training. 
            # layerIdMat = eye(numNeurons,'like',x);
            # layeri.backprop.store.idMat = layerIdMat; % store id-matrix
            # additionally, we store the indices of the generators
            # corresponding the approximations error of the activation
            # layer. The activation layer simply (i) multiplies the 
            # id-matrix with the vector containing approximation error 
            # and (ii) copies the new generators in the correct spot in
            # the generator matrix.
            layerNumApproxErr = min(numApproxErr, numNeurons)
            layeri.backprop['store']['approxErrGenIds'] = list(range(1 + numGen, 
                                                                    numGen + layerNumApproxErr + 1))  # 1-based indexing
            # add a generator for each neuron to store the approx. error
            numGen = numGen + min(numApproxErr, numNeurons)
        else:
            # no approximation error are stored
            layeri.backprop['store']['approxErrGenIds'] = []
    elif hasattr(layeri, '__class__') and 'nnElementwiseAffineLayer' in str(layeri.__class__):
        # Does not change the number of neurons.
        return numGen, numNeurons
    else:
        # Update the number of neurons.
        _, numNeurons = layeri.getNumNeurons()
    
    return numGen, numNeurons
