"""
explain - compute a minimal abductive explanation

Description:
    Compute a minimal abductive explanation

Syntax:
    [idxFreedFeats, featOrder, timesPerFeat] = explain(x, target, epsilon, verbose, method, featOrderMethod, refineMethod, inputSize, refineSteps, bucketType, delta, timeout)

Inputs:
    x - Input point
    target - Target label/class
    epsilon - Noise radius
    verbose - Verbose output
    method - Explanation method
    featOrderMethod - Feature ordering method
    refineMethod - Refinement method
    inputSize - Input size specification
    refineSteps - Refinement steps
    bucketType - Bucket type
    delta - Delta parameter
    timeout - Timeout value

Outputs:
    idxFreedFeats - Indices of freed features
    featOrder - Feature order
    timesPerFeat - Times per feature

Example:
    idxFreedFeats, featOrder, timesPerFeat = nn.explain(x, target, 0.1, verbose=True)

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2022 (polish)
Last update:   23-November-2022 (polish)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import time
from typing import Any, List, Tuple
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def explain(self, x: np.ndarray, target: Any, epsilon: float, **kwargs) -> Tuple[List[int], List[int], List[float]]:
    """
    Compute a minimal abductive explanation
    
    Args:
        x: Input point
        target: Target label/class
        epsilon: Noise radius
        **kwargs: Additional arguments including verbose, method, featOrderMethod, etc.
        
    Returns:
        Tuple of (idxFreedFeats, featOrder, timesPerFeat) results
    """
    
    def _aux_checkVerifiability(x: np.ndarray, epsilon: float, featOrder: np.ndarray, 
                               inputSize: List[int], delta: float) -> bool:
        """Check if all features can be freed initially"""
        X = _aux_initInputSet(x, epsilon, featOrder, inputSize)
        Y = self.evaluate(X)
        return _aux_checkSpecs(Y, delta)

    def _aux_initInputSet(x: np.ndarray, epsilon: float, idxFree: np.ndarray, 
                         inputSize: List[int]):
        """Initialize input set with freed features"""
        from cora_python.contSet.interval import Interval
        from cora_python.contSet.zonotope import Zonotope
        
        # Create interval from input point
        X = Interval(x)
        
        if inputSize is None or inputSize == 'all':
            # allow any value for free features
            for idx in idxFree:
                # Convert to 0-based indexing for Python
                idx_py = int(idx) - 1 if idx > 0 else int(idx)
                if 0 <= idx_py < len(X):
                    X[idx_py] = X[idx_py] + Interval(-epsilon, epsilon)  # pixel +/- epsilon
        else:
            # allow any value for free pixels
            X = X.reshape([inputSize[0]*inputSize[1], inputSize[2]])
            for idx in idxFree:
                # Convert to 0-based indexing for Python
                idx_py = int(idx) - 1 if idx > 0 else int(idx)
                if 0 <= idx_py < X.shape[0]:
                    X[idx_py, :] = X[idx_py, :] + Interval(-epsilon, epsilon)  # pixel +/- epsilon
            X = X.reshape(-1, 1)
        
        return Zonotope(X)  # convert to zonotope

    def _aux_checkSpecs(Y, delta: float) -> bool:
        """Check specifications for the output set"""
        # Check specification, works for set and numeric input
        
        if isinstance(Y, np.ndarray):
            # numeric input
            if Y.ndim == 0 or Y.size == 1:
                # regression 
                # (unable to determine deviation here)
                return True
            else:
                # classification
                ys = Y
                us = np.max(ys, axis=0)
                idxSpurious = np.where(np.max(us, axis=0) > 0)[0]
                return len(idxSpurious) == 0
        else:
            # set input
            from cora_python.contSet.interval import Interval
            
            Y_interval = Interval(Y)
            if Y_interval.dim == 1:
                # regression
                return Y_interval.rad <= delta
            else:
                # classification
                return np.all(Y_interval.sup <= 0)

    def _aux_appendLogitDifferenceLayer(nn, label: int):
        """Append a layer to output the logit difference"""
        from nn.layers.linear.nnLinearLayer import nnLinearLayer
        
        # Create weight matrix for logit difference
        W_argmax = np.eye(nn.neurons_out)
        W_argmax[:, label] = W_argmax[:, label] - 1
        
        # Append the layer
        nn.layers.append(nnLinearLayer(W_argmax, np.zeros((nn.neurons_out, 1)), 'logit difference'))
        
        # Get neural network in normal form
        nn = nn.getNormalForm(nn)
        return nn

    def _aux_runVerification(nn_abs, x: np.ndarray, target: Any, epsilon: float, 
                            verbose: bool, method: str, refineMethod: str, 
                            idxFreedFeats_i: List[int], inputSize: List[int], 
                            refineSteps: List[float], bucketType: str, delta: float):
        """Run verification for the current set of freed features"""
        
        # construct input set
        X = _aux_initInputSet(x, epsilon, np.array(idxFreedFeats_i), inputSize)
        
        # run verification
        if method == 'standard':
            if verbose:
                print('Using original network..')
            
            # compute output set
            Y = self.evaluate(X)
            
            # check specification (classification)
            isVerified = _aux_checkSpecs(Y, delta)
            return isVerified, nn_abs
            
        elif method == 'abstract+refine':
            # This would require implementing the abstract+refine method
            # For now, we'll use the standard method
            if verbose:
                print('Using original network (abstract+refine not fully implemented)..')
            
            # compute output set
            Y = self.evaluate(X)
            
            # check specification
            isVerified = _aux_checkSpecs(Y, delta)
            return isVerified, nn_abs
        
        else:
            raise CORAerror('CORA:wrongValue', f"Unknown method: {method}")
    
    # parse inputs
    verbose = kwargs.get('verbose', False)
    method = kwargs.get('method', 'standard')
    featOrderMethod = kwargs.get('featOrderMethod', 'sensitivity')
    refineMethod = kwargs.get('refineMethod', 'standard')
    inputSize = kwargs.get('inputSize', None)
    refineSteps = kwargs.get('refineSteps', [])
    bucketType = kwargs.get('bucketType', 'static')
    delta = kwargs.get('delta', 0.1)
    timeout = kwargs.get('timeout', 300.0)
    
    # validate inputs
    if inputSize is None:
        inputSize = [x.shape[0], 1, 1]  # Default to 1D input
    
    # init plotting (simplified - would need proper plotting library)
    if verbose:
        print(f"Starting explanation for input with shape {x.shape}")
    
    # have logic difference as output for easier specification check 
    # (for classification tasks: n_out > 1, otherwise regression)
    if self.neurons_out is not None and self.neurons_out > 1:
        # Append logit difference layer
        self = _aux_appendLogitDifferenceLayer(self, target)
    
    # get processing order and init freed features
    if isinstance(featOrderMethod, (list, np.ndarray)):
        featOrder = featOrderMethod
    else:
        featOrder = self.getInputNeuronOrder(featOrderMethod, x, inputSize)
    
    idxFreedFeats = []
    timesPerFeat = [np.nan] * len(featOrder)
    
    # check if all features can be freed
    timerVal = time.time()
    if verbose:
        print('Checking if all features can be freed..')
    
    # Check if all features can be freed initially
    isVerified = _aux_checkVerifiability(x, epsilon, featOrder, inputSize, delta)
    t = time.time() - timerVal
    if verbose:
        print(f'Time to compute initial check: {t:.4f}.')
    
    if isVerified:
        print('All features can be freed.')
        idxFreedFeats = featOrder.tolist()
        timesPerFeat = [0.0] * len(featOrder)
        # Convert to numpy arrays to match MATLAB behavior
        return np.array(idxFreedFeats), np.array(featOrder), np.array(timesPerFeat)
    
    # init abstract network (used for some methods)
    nn_abs = None
    
    # iteratively free features
    for i in range(len(featOrder)):
        if verbose:
            print('---')
            print(f'Freeing input feature: {i+1}/{len(featOrder)} ({featOrder[i]}):')
        
        # temporarily add feature i to freed features
        idxFreedFeats_i = idxFreedFeats + [featOrder[i]]
        
        # run verification
        timerVal = time.time()
        isVerified, nn_abs = _aux_runVerification(nn_abs, x, target, epsilon, verbose, method, 
                                                  refineMethod, idxFreedFeats_i, inputSize, 
                                                  refineSteps, bucketType, delta)
        timesPerFeat[i] = time.time() - timerVal
        if verbose:
            print(f'Elapsed time: {timesPerFeat[i]:.4f}.')
        
        if isVerified:  # verified
            # permanently add feature i to freed features
            idxFreedFeats = idxFreedFeats_i
            
            # verbose output
            if verbose:
                print(f'Input feature {featOrder[i]} was freed.')
        else:  # could not verify freed features
            if verbose:
                print(f'Input feature {featOrder[i]} cannot be freed.')
        
        # check if freeing next feature would exceed timeout
        if i < len(featOrder) - 1 and (i+1) * np.mean(timesPerFeat[:i+1]) > timeout:
            print('Timeout!')
            break
    
    # Convert to numpy arrays to match MATLAB behavior
    return np.array(idxFreedFeats), np.array(featOrder), np.array(timesPerFeat)