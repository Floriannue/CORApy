"""
validateNNoptions - checks input and sets default values for the
   options.nn struct for the neuralNetwork/evaluate function.

Description:
    Checks input and sets default values for the options.nn struct 
    for the neuralNetwork/evaluate function.

Syntax:
    options = validateNNoptions(options,setTrainFields)

Inputs:
    options - struct (see neuralNetwork/evaluate)
    setTrainFields - bool, decide if training parameters should be
       validated (see neuralNetwork/train)

Outputs:
    options - updated options

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/evaluate

Authors:       Tobias Ladner, Lukas Koller
Written:       29-November-2022
Last update:   16-February-2023 (poly_method)
                01-March-2023 (backprop)
                21-February-2024 (merged options.nn)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Dict, Any, Union, Callable
import numpy as np


def validateNNoptions(options: Dict[str, Any], set_train_fields: bool = False) -> Dict[str, Any]:
    """
    Check input and set default values for the options.nn struct.
    
    Args:
        options: Options dictionary
        set_train_fields: Whether to validate training parameters
        
    Returns:
        Updated options dictionary
    """
    # Default values
    default_fields = {
        'bound_approx': True,
        'poly_method': lambda opts: aux_defaultPolyMethod(opts),
        'num_generators': None,
        'max_gens_post': lambda opts: opts.get('num_generators', 100) * 100,
        'add_approx_error_to_GI': False,
        'plot_multi_layer_approx_info': False,
        'reuse_bounds': False,
        'max_bounds': 5,
        'do_pre_order_reduction': True,
        'remove_GI': lambda opts: not opts.get('add_approx_error_to_GI', False),
        'force_approx_lin_at': float('inf'),
        'propagate_bounds': lambda opts: opts.get('reuse_bounds', False),
        'sort_exponents': False,
        'maxpool_type': 'project',
        'order_reduction_sensitivity': False,
        'use_approx_error': True,
        'train': {'backprop': False},
        'interval_center': False,
        # gnn
        'graph': None,  # graph()
        'idx_pert_edges': [],
        'invsqrt_order': 1,
        'invsqrt_use_approx_error': True,
        # evaluateZonotopeBatch
        'use_approx_error': True,  # use approximation error
        'interval_center': False,  # the center stores approximation errors as an interval
        # training
        'train': {'backprop': False}  # use approximation error
    }
    
    # Default training parameter values
    default_train_fields = {
        # general
        'use_gpu': aux_isGPUavailable(),
        'optim': 'nnAdamOptimizer',  # optimizer
        'max_epoch': 10,  # maximum number of training epochs
        'mini_batch_size': 64,  # batch size
        'loss': 'mse',  # loss type
        'lossFun': lambda opts: lambda t, y: 0,  # custom loss
        'lossDer': lambda opts: lambda t, y: 0,  # gradient of custom loss
        'backprop': True,  # enable backpropagation
        'shuffle_data': 'never',  # shuffle the training data
        'early_stop': float('inf'),  # early stopping when val loss is not decreasing
        'lr_decay': 1,  # factor for learning rate decay
        'lr_decay_epoch': [],  # epochs for learning rate decay
        'val_freq': 50,  # validation frequency
        'print_freq': 50,  # verbose training output printing frequency
        # Robust training parameters
        'noise': 0,  # perturbation radius
        'input_space_inf': 0,  # bounds of the input space
        'input_space_sup': 1,
        'warm_up': 0,  # number of 'warm-up' training epochs (noise=0)
        'ramp_up': 0,  # noise=0 is linearly increase from 'warm-up' to 'ramp-up'
        'method': 'point',  # training method
        'kappa': 1/2,  # IBP weighting factor
        'lambda': 0,  # SABR & TRADES weighting factor
        # attack
        'pgd_iterations': 0,
        'pgd_stepsize': 0.01,
        'pgd_stepsize_decay': 1,
        'pgd_stepsize_decay_iter': [],
        # set-based training
        'volume_heuristic': 'interval',
        'tau': 0,  # weighting factor
        'zonotope_weight_update': 'center',  # compute weight update
        'exact_backprop': False,  # exact gradient computations of image enc
        'num_approx_err': float('inf'),  # maximum number of approximation errors per nonlinear layer
        'num_init_gens': float('inf'),  # maximum number of input generators
        'init_gens': 'l_inf'  # type of input generators
    }
    
    # Check if any nn options are given
    if 'nn' not in options:
        options['nn'] = {}
    
    # Set default value of fields if required
    options['nn'] = setDefaultFields(options['nn'], default_fields)
    
    # Set default training parameters
    if set_train_fields:
        if 'train' not in options['nn']:
            options['nn']['train'] = {}
        options['nn']['train'] = setDefaultFields(options['nn']['train'], default_train_fields)
    
    # Check fields
    # Note: In Python, we'll skip the detailed field validation for now
    # as it would require implementing the full CORA error checking system
    
    return options


# Auxiliary functions -----------------------------------------------------

def setDefaultFields(options_nn: Dict[str, Any], default_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set default fields in options.nn if they don't exist.
    
    Args:
        options_nn: Options.nn dictionary
        default_fields: Default field values
        
    Returns:
        Updated options.nn dictionary
    """
    for field, default_value in default_fields.items():
        if field not in options_nn:
            if callable(default_value):
                # Handle callable defaults
                options_nn[field] = default_value(options_nn)
            else:
                options_nn[field] = default_value
    
    return options_nn


def aux_isGPUavailable() -> bool:
    """
    Check if GPU is available.
    
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        # Try to import torch to check for GPU
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            # Try to import tensorflow to check for GPU
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            # No GPU libraries available
            return False


def aux_defaultPolyMethod(options: Dict[str, Any]) -> str:
    """
    Get default poly method based on options.
    
    Args:
        options: Options dictionary
        
    Returns:
        Default poly method string
    """
    if 'train' in options:
        return 'bounds'
    else:
        return 'regression'
