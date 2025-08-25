"""
visLossLandscape - Visualize the loss landscape of a neural network.

Syntax:
    [alpha,beta,lossCenter,lossVol,lossTotal] =...
       nn.visLossLandscape(x,t,options,numSamples);
    figure;
    contour(alpha,beta,lossCenter,10); # contour plot of loss landscape

Inputs:
    obj - neural network
    x - test inputs; i-th input x[:,i]
    t - test targets; i-th target t[:,i]
    options - training parameters (see neuralNetwork.train),
           needed to pass information how loss is computed (training
           perturbation radius, scaling of volume in loss, etc.)
    numSamples - number of samples along gradient axis (default=20)

Outputs:
    alpha - values along the first axis; represents a gradient direction 
    beta - values along the second axis; represents a gradient direction 
    lossCenter - matrix with loss values, center error
    lossVol - matrix with loss values, volume heuristic error
    lossTotal - matrix with loss values, total error
    
References:
    [1] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, et al. 
        "Visualizing the loss landscape of neural nets". In NeurIPS, 
        pages 6391–6401, 2018. 4, 5

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Lukas Koller
Written:       21-June-2023
Last update:   02-August-2023 (adapted variable names)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from .neuralNetwork import NeuralNetwork


def visLossLandscape(obj: NeuralNetwork, x: np.ndarray, t: np.ndarray, 
                    options: Dict[str, Any], numSamples: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Visualize the loss landscape of a neural network.
    
    Args:
        obj: neural network
        x: test inputs; i-th input x[:,i]
        t: test targets; i-th target t[:,i]
        options: training parameters (see neuralNetwork.train)
        numSamples: number of samples along gradient axis (default=20)
        
    Returns:
        alpha: values along the first axis; represents a gradient direction 
        beta: values along the second axis; represents a gradient direction 
        lossCenter: matrix with loss values, center error
        lossVol: matrix with loss values, volume heuristic error
        lossTotal: matrix with loss values, total error
    """
    # parse input - equivalent to narginchk(4,5)
    # numSamples is already handled by default parameter
    
    # validate input - equivalent to inputArgsCheck
    if not hasattr(obj, '__class__') or 'neuralNetwork' not in str(obj.__class__):
        raise ValueError("First argument must be a neuralNetwork object")
    
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numeric array")
    
    if not isinstance(t, np.ndarray):
        raise ValueError("t must be a numeric array")
    
    if not isinstance(options, dict):
        raise ValueError("options must be a dictionary")
    
    if not isinstance(numSamples, (int, np.integer)):
        raise ValueError("numSamples must be numeric")
    
    # use the neuralNetwork.train function to compute the loss
    # setup training params
    if 'nn' not in options:
        options['nn'] = {}
    if 'train' not in options['nn']:
        options['nn']['train'] = {}
    
    # Create nnSGDOptimizer instance
    from ...nn.optim.nnSGDOptimizer import nnSGDOptimizer
    options['nn']['train']['optim'] = nnSGDOptimizer(0)
    options['nn']['train']['max_epoch'] = 1
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # sample random weight directions \delta and \eta
    delta = _aux_generateRandomNormalizedWeights(obj)
    eta = _aux_generateRandomNormalizedWeights(obj)
    
    # scales for plot
    alpha = np.linspace(-1.5, 1.5, numSamples)
    beta = np.linspace(-1.5, 1.5, numSamples)
    
    # store loss
    lossTotal = np.zeros((len(alpha), len(beta)))
    lossCenter = np.zeros((len(alpha), len(beta)))
    lossVol = np.zeros((len(alpha), len(beta)))
    
    # compute losses
    for i in range(len(alpha)):
        for j in range(len(beta)):
            # create copy of the neural network to modify the weights
            nnTest = obj.copyNeuralNetwork()
            
            for k in range(len(nnTest.layers)):
                # extract original weights and biases
                layerk = obj.layers[k]
                if (hasattr(layerk, '__class__') and 
                    ('nnLinearLayer' in str(layerk.__class__) or 'nnConv2dLayer' in str(layerk.__class__))):
                    
                    Wk = layerk.W
                    bk = layerk.b
                    
                    # compute weight and bias updates
                    deltak = delta[k]
                    etak = eta[k]
                    
                    if deltak is not None and etak is not None:
                        dWk = alpha[i] * deltak[:, :-1] + beta[j] * etak[:, :-1]
                        dbk = alpha[i] * deltak[:, -1] + beta[j] * etak[:, -1]
                        
                        # update weight matrix and bias
                        nnTest.layers[k].W = Wk + dWk
                        nnTest.layers[k].b = bk + dbk
            
            # compute loss for input data (just validation data)
            try:
                lossij = nnTest.train(x, t, None, None, options, False)
                lossCenter[i, j] = lossij['center']  # lossij.valCenter;
                lossVol[i, j] = lossij['vol']  # lossij.valVol;
                lossTotal[i, j] = lossij['total']  # lossij.valTotal;
            except Exception as e:
                # If training fails, set loss to infinity
                lossCenter[i, j] = float('inf')
                lossVol[i, j] = float('inf')
                lossTotal[i, j] = float('inf')
    
    return alpha, beta, lossCenter, lossVol, lossTotal


def _aux_generateRandomNormalizedWeights(nn: NeuralNetwork) -> List[Optional[np.ndarray]]:
    """
    Auxiliary function to generate random normalized weights
    
    Args:
        nn: neural network
        
    Returns:
        List of normalized weight matrices for each layer
    """
    # copy neural network
    nnCopy = nn.copyNeuralNetwork()
    # initialize random weights
    # nnCopy.initWeights('glorot');
    
    numLayers = len(nnCopy.layers)
    # sample random weight directions \delta and \eta
    delta = [None] * numLayers
    
    for i in range(numLayers):
        layeri = nn.layers[i]
        if (hasattr(layeri, '__class__') and 
            ('nnLinearLayer' in str(layeri.__class__) or 'nnConv2dLayer' in str(layeri.__class__))):
            
            # extract weight matrix and bias
            Wi = layeri.W
            bi = layeri.b
            m, n = Wi.shape
            
            # sample random directions with appropriate size + normalize
            # equivalent to mvnrnd(zeros(m,1),eye(m),n)'
            deltaWi = np.random.multivariate_normal(np.zeros(m), np.eye(m), n).T
            deltaWi = np.linalg.norm(Wi, 'fro') / np.linalg.norm(deltaWi, 'fro') * deltaWi  # normalize
            
            # equivalent to mvnrnd(zeros(m,1),eye(m),1)'
            deltabi = np.random.multivariate_normal(np.zeros(m), np.eye(m), 1).T
            deltabi = np.linalg.norm(bi, 'fro') / np.linalg.norm(deltabi, 'fro') * deltabi  # normalize
            
            delta[i] = np.hstack([deltaWi, deltabi])
            # delta[i] = [Wi bi];
        else:
            delta[i] = None
    
    return delta


def _aux_generateNormalizedLossGradient(nn: NeuralNetwork, x: np.ndarray, t: np.ndarray, trainParams: Dict[str, Any]) -> List[Optional[np.ndarray]]:
    """
    Auxiliary function to generate normalized loss gradient
    
    Args:
        nn: neural network
        x: input data
        t: target data
        trainParams: training parameters
        
    Returns:
        List of normalized gradients for each layer
    """
    v0, _ = x.shape
    
    numInitGens = trainParams.get('num_init_gens', v0)
    numApproxErr = trainParams.get('num_approx_err', 0)
    
    # copy neural network
    nnCopy = nn.copyNeuralNetwork()
    # initialize network for batch propagation
    nnCopy.prepareForZonoBatchEval(x, min(v0, numInitGens), numApproxErr)
    
    # init output
    numLayers = len(nnCopy.layers)
    delta = [None] * numLayers
    
    # propagate inputs through the network
    nnCopy.train(x, t, None, None, trainParams, False)
    
    # extract gradient from network
    for i in range(numLayers):
        layeri = nnCopy.layers[i]
        if (hasattr(layeri, '__class__') and 
            ('nnLinearLayer' in str(layeri.__class__) or 'nnConv2dLayer' in str(layeri.__class__))):
            
            # extract weight matrix and bias
            Wi = layeri.W
            bi = layeri.b
            
            # extract gradient for weight matrix and bias
            if hasattr(layeri, 'backprop') and hasattr(layeri.backprop, 'grad'):
                deltaWi = layeri.backprop['grad']['W']
                deltaWi = np.linalg.norm(Wi, 'fro') / np.linalg.norm(deltaWi, 'fro') * deltaWi  # normalize
                
                deltabi = layeri.backprop['grad']['b']
                deltabi = np.linalg.norm(bi, 'fro') / np.linalg.norm(deltabi, 'fro') * deltabi  # normalize
                
                delta[i] = np.hstack([deltaWi, deltabi])
            else:
                delta[i] = None
        else:
            delta[i] = None
    
    return delta


def _aux_convexCombOfNetworkParams(delta1: List[Optional[np.ndarray]], 
                                  delta2: List[Optional[np.ndarray]], 
                                  lambda_val: float) -> List[Optional[np.ndarray]]:
    """
    Auxiliary function to compute convex combination of network parameters
    
    Args:
        delta1: first set of parameters
        delta2: second set of parameters
        lambda_val: combination parameter
        
    Returns:
        Convex combination of parameters
    """
    numLayers = len(delta1)
    delta = [None] * numLayers
    
    for i in range(numLayers):
        if delta1[i] is not None and delta2[i] is not None:
            delta[i] = lambda_val * delta1[i] + (1 - lambda_val) * delta2[i]
        else:
            delta[i] = None
    
    return delta
