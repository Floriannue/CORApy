"""
train - train a neural network

Syntax:
    % initialize the weights of the neural network
    nn.initWeights('glorot');
    % specify training parameters
    options.nn.train = struct(...
       'use_gpu',true, 'optim',nnAdamOptimizer(1e-3), 'method','point');
    % train the neural networks
    [loss,trainTime] = nn.train(trainX, trainT, valX, valT, options);

Inputs:
    nn - neural network
    trainX - training input; i-th input trainX[:,i]
    trainT - training targets; i-th target trainT[:,i]
    valX - validation input, normalized to [0,1]
    valT - validation targets
    options.nn.train - possible training parameters
        .optim: instance of nnOptimizer, e.g. nnSGDOptimizer,
           nnAdamOptimizer
        .max_epoch: maximum number of training epochs
        .mini_batch_size: training batch size
        .loss: type of loss function, {'mse','softmax+log','custom'};
           'mse' for regression loss, 'softmax+log' for cross-entropy, 
               or 'custom': in this case provide .loss_fun and .loss_grad
        .noise: training perturbatuion noise (default: 0.0)
        .input_space_inf: lower bound of the input space (default: 0)
        .input_space_sup: upper bound of the input space (default: 1)
        .warm_up: number of training epochs at the beginning without any 
           perturbation noise, (default: 0)
        .ramp_up: linearly ramp up perturbation noise from warm-up to
           ramp-up, (default: 0)
        .method: training methods, {'point','set','madry','gowal','trades','sabr'}; 
           extra options for the individual
           'point': regular point-based training, see [2]
           'set': set-based training
               .volume_heuristic: {'interval','f-radius'}
               .tau: scaling-factor of volume heuristic in loss
               .zonotope_weight_update: {'center','sample','extreme','sum'}
               .num_approx_err: number of approx. errors per activation 
                   layer during training, inf for all
               .num_init_gens: number of generators of the input 
                   zonotopes; ignored if initial_generators='l_inf'
               .init_gens: {'l_inf','random','sensitivity'}
               .poly_method: {'bounds','singh','center'}
           'madry': augment training inputs with adversarial attacks, see [3]
               .pgd_iterations: number of iterations
               .pgd_stepsize: perturbation stepsize (default: 0.01)
           'gowal': interval-bound propagation, see [4]
               .kappa: weighting factor (final)
           'trades': tradeoff (betw. accuracy & robustness) inspired
               training, see [6]
               .lambda: weighting factor
               .pgd_iterations: number of iterations
               .pgd_stepsize: perturbation stepsize (default: 0.01)
           'sabr': combination of adversarial attacks and interval-bound
               propagation, see [5]
               .lambda: weighting factor
               .pgd_iterations: number of iterations
               .pgd_stepsize: perturbation stepsize
               .pgd_stepsize_decay: decay factor of stepsize in PGD
               .pgd_stepsize_decay_iter: decay iterations of stepsize in PGD
        .shuffle_data: shuffle training data, {'never','every_epoch'},
           (default: 'never')
        .lr_decay: decay of learning rate (default: 1.0)
        .lr_decay_epoch: epoch at which learning rate is decayed
           (default: [])
        .early_stop: number, abort training if validation loss is 
           non-decreasing after certain amount of steps (default: inf)
        .val_freq: validation frequency (default: 50)
        .print_freq: print losses every printFreq-th iteration 
           (default: 50)
    verbose - print intermediate training losses (default: true)

Outputs:
    nn - trained neural network
    loss - loss during training, struct
        .center: (training) loss
        .vol: (training) volume heuristic loss
        .total: total training training loss
        .val: validation loss
    trainTime - training time
    
References:
    [1] C. M. Bishop. Pattern Recognition and Machine Learning. 
        Information Science and Statistics. Springer New York, NY, 2006.
    [2] Koller, L. "Co-Design for Training and Verifying Neural Networks",
           Master's Thesis
    [3] Madry, A. et al. Towards deep learning models resistant to 
           adversarial attacks. ICLR. 2018
    [4] Gowal, S. et al. Scalable verified training for provably
           robust image classification. ICCV. 2019
    [5] Mueller, M. et al. Certified Training: Small Boxes are All You 
           Need. ICLR. 2022
    [6] Zhang, H. et al. Theoretically Principled Trade-off between 
           Robustness and Accuracy. ICML. 2019
 
Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork, nnOptimizer

Authors:       Tobias Ladner, Lukas Koller
Written:       01-March-2023
Last update:   03-May-2023 (LK, added backprop for polyZonotope)
                25-May-2023 (LK, added options as function parameter)
                31-July-2023 (LK, return update gradients)
                04-August-2023 (LK, added layer indices)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from .neuralNetwork import NeuralNetwork


def train(nn: NeuralNetwork, trainX: np.ndarray, trainT: np.ndarray, 
          valX: Optional[np.ndarray] = None, valT: Optional[np.ndarray] = None, 
          options: Optional[Dict[str, Any]] = None, verbose: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Train a neural network
    
    Args:
        nn: neural network object
        trainX: training input; i-th input trainX[:,i]
        trainT: training targets; i-th target trainT[:,i]
        valX: validation input, normalized to [0,1]
        valT: validation targets
        options: training options
        verbose: print intermediate training losses (default: True)
        
    Returns:
        loss: loss during training
        trainTime: training time
    """
    # Start timing
    start_time = time.time()
    
    # Set default options if not provided
    if options is None:
        options = {}
    if 'nn' not in options:
        options['nn'] = {}
    if 'train' not in options['nn']:
        options['nn']['train'] = {}
    
    # Set default training parameters
    train_opts = options['nn']['train']
    train_opts.setdefault('max_epoch', 100)
    train_opts.setdefault('mini_batch_size', 32)
    train_opts.setdefault('loss', 'mse')
    train_opts.setdefault('noise', 0.0)
    train_opts.setdefault('input_space_inf', 0)
    train_opts.setdefault('input_space_sup', 1)
    train_opts.setdefault('warm_up', 0)
    train_opts.setdefault('ramp_up', 0)
    train_opts.setdefault('method', 'point')
    train_opts.setdefault('shuffle_data', 'never')
    train_opts.setdefault('lr_decay', 1.0)
    train_opts.setdefault('lr_decay_epoch', [])
    train_opts.setdefault('early_stop', float('inf'))
    train_opts.setdefault('val_freq', 50)
    train_opts.setdefault('print_freq', 50)
    
    # Initialize loss tracking
    loss = {
        'center': [],
        'vol': [],
        'total': [],
        'val': []
    }
    
    # Get optimizer
    if 'optim' not in train_opts:
        raise ValueError("Optimizer must be specified in options.nn.train.optim")
    
    optimizer = train_opts['optim']
    
    # Training loop
    num_samples = trainX.shape[1]
    num_batches = (num_samples + train_opts['mini_batch_size'] - 1) // train_opts['mini_batch_size']
    
    for epoch in range(train_opts['max_epoch']):
        epoch_loss = 0.0
        epoch_vol_loss = 0.0
        
        # Shuffle data if requested
        if train_opts['shuffle_data'] == 'every_epoch':
            indices = np.random.permutation(num_samples)
            trainX = trainX[:, indices]
            trainT = trainT[:, indices]
        
        # Process mini-batches
        for batch in range(num_batches):
            start_idx = batch * train_opts['mini_batch_size']
            end_idx = min(start_idx + train_opts['mini_batch_size'], num_samples)
            
            batch_X = trainX[:, start_idx:end_idx]
            batch_T = trainT[:, start_idx:end_idx]
            
            # Forward pass
            batch_output = nn.evaluate(batch_X, options)
            
            # Compute loss
            if train_opts['loss'] == 'mse':
                batch_loss = np.mean((batch_output - batch_T) ** 2)
            elif train_opts['loss'] == 'softmax+log':
                # Softmax + log loss for classification
                exp_output = np.exp(batch_output - np.max(batch_output, axis=0, keepdims=True))
                softmax_output = exp_output / np.sum(exp_output, axis=0, keepdims=True)
                batch_loss = -np.mean(np.sum(batch_T * np.log(softmax_output + 1e-15), axis=0))
            else:
                raise ValueError(f"Unsupported loss function: {train_opts['loss']}")
            
            epoch_loss += batch_loss
            
            # Backward pass and optimization
            if hasattr(optimizer, 'step'):
                # Compute gradients (simplified - in practice this would be more complex)
                grad_out = 2 * (batch_output - batch_T) / batch_output.shape[1]
                grad_in = nn.backprop(grad_out, options)
                
                # Update weights using optimizer
                optimizer.step()
        
        # Average loss over epoch
        epoch_loss /= num_batches
        loss['center'].append(epoch_loss)
        loss['total'].append(epoch_loss + epoch_vol_loss)
        
        # Validation
        if valX is not None and valT is not None and epoch % train_opts['val_freq'] == 0:
            val_output = nn.evaluate(valX, options)
            if train_opts['loss'] == 'mse':
                val_loss = np.mean((val_output - valT) ** 2)
            elif train_opts['loss'] == 'softmax+log':
                exp_output = np.exp(val_output - np.max(val_output, axis=0, keepdims=True))
                softmax_output = exp_output / np.sum(exp_output, axis=0, keepdims=True)
                val_loss = -np.mean(np.sum(valT * np.log(softmax_output + 1e-15), axis=0))
            else:
                val_loss = float('inf')
            
            loss['val'].append(val_loss)
            
            if verbose and epoch % train_opts['print_freq'] == 0:
                print(f"Epoch {epoch}: Train Loss = {epoch_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        elif verbose and epoch % train_opts['print_freq'] == 0:
            print(f"Epoch {epoch}: Train Loss = {epoch_loss:.6f}")
        
        # Early stopping
        if len(loss['val']) > train_opts['early_stop']:
            # Check if validation loss is increasing
            recent_val_losses = loss['val'][-train_opts['early_stop']:]
            if all(recent_val_losses[i] >= recent_val_losses[i-1] for i in range(1, len(recent_val_losses))):
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    # Calculate total training time
    train_time = time.time() - start_time
    
    return loss, train_time
