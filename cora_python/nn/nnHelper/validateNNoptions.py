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
from .setDefaultFields import setDefaultFields


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
    default_fields = [
        ['bound_approx', True],
        ['poly_method', lambda opts: aux_defaultPolyMethod(opts)],
        ['num_generators', None],
        ['max_gens_post', lambda opts: (opts.get('num_generators') or 100) * 100 if opts else 10000],
        ['add_approx_error_to_GI', False],
        ['plot_multi_layer_approx_info', False],
        ['reuse_bounds', False],
        ['max_bounds', 5],
        ['do_pre_order_reduction', True],
        ['remove_GI', lambda opts: not opts.get('add_approx_error_to_GI', False)],
        ['force_approx_lin_at', float('inf')],
        ['propagate_bounds', lambda opts: opts.get('reuse_bounds', False)],
        ['sort_exponents', False],
        ['maxpool_type', 'project'],
        ['order_reduction_sensitivity', False],
        ['use_approx_error', True],
        ['train', {'backprop': False}],
        ['interval_center', False],
        # gnn
        ['graph', None],  # graph()
        ['idx_pert_edges', []],
        ['invsqrt_order', 1],
        ['invsqrt_use_approx_error', True],
        # evaluateZonotopeBatch
        ['use_approx_error', True],  # use approximation error
        ['interval_center', False],  # the center stores approximation errors as an interval
        # training
        ['train', {'backprop': False}]  # use approximation error
    ]
    
    # Default training parameter values
    default_train_fields = [
        # general
        ['use_gpu', aux_isGPUavailable()],
        ['optim', 'nnAdamOptimizer'],  # optimizer
        ['max_epoch', 10],  # maximum number of training epochs
        ['mini_batch_size', 64],  # batch size
        ['loss', 'mse'],  # loss type
        ['lossFun', lambda opts: lambda t, y: 0],  # custom loss
        ['lossDer', lambda opts: lambda t, y: 0],  # gradient of custom loss
        ['backprop', True],  # enable backpropagation
        ['shuffle_data', 'never'],  # shuffle the training data
        ['early_stop', float('inf')],  # early stopping when val loss is not decreasing
        ['lr_decay', 1],  # factor for learning rate decay
        ['lr_decay_epoch', []],  # epochs for learning rate decay
        ['val_freq', 50],  # validation frequency
        ['print_freq', 50],  # verbose training output printing frequency
        # Robust training parameters
        ['noise', 0],  # perturbation radius
        ['input_space_inf', 0],  # bounds of the input space
        ['input_space_sup', 1],
        ['warm_up', 0],  # number of 'warm-up' training epochs (noise=0)
        ['ramp_up', 0],  # noise=0 is linearly increase from 'warm-up' to 'ramp-up'
        ['method', 'point'],  # training method
        ['kappa', 1/2],  # IBP weighting factor
        ['lambda', 0],  # SABR & TRADES weighting factor
        # attack
        ['pgd_iterations', 0],
        ['pgd_stepsize', 0.01],
        ['pgd_stepsize_decay', 1],
        ['pgd_stepsize_decay_iter', []],
        # set-based training
        ['volume_heuristic', 'interval'],
        ['tau', 0],  # weighting factor
        ['zonotope_weight_update', 'center'],  # compute weight update
        ['exact_backprop', False],  # exact gradient computations of image enc
        ['num_approx_err', float('inf')],  # maximum number of approximation errors per nonlinear layer
        ['num_init_gens', float('inf')],  # maximum number of input generators
        ['init_gens', 'l_inf']  # type of input generators
    ]
    
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
    # Note: In Python, we'll implement the validation checks but skip CORA-specific error handling
    # as it would require implementing the full CORA error checking system
    
    # bound_approx
    aux_checkFieldClass(options['nn'], 'bound_approx', ['bool', 'str'], 'options')
    if isinstance(options['nn']['bound_approx'], str):
        aux_checkFieldStr(options['nn'], 'bound_approx', ['sample'], 'options')
        print("Warning: Choosing Bound estimation '{}' does not lead to safe verification!".format(
            options['nn']['bound_approx']))
    
    # poly_method
    validPolyMethod = ['regression', 'ridgeregression', 'throw-catch', 'taylor', 'singh', 'bounds', 'center']
    aux_checkFieldStr(options['nn'], 'poly_method', validPolyMethod, 'options')
    
    # num_generators
    aux_checkFieldClass(options['nn'], 'num_generators', ['int', 'float', 'type(None)'], 'options')
    
    # add_approx_error_to_GI
    aux_checkFieldClass(options['nn'], 'add_approx_error_to_GI', ['bool'], 'options')
    
    # plot_multi_layer_approx_info
    aux_checkFieldClass(options['nn'], 'plot_multi_layer_approx_info', ['bool'], 'options')
    
    # reuse_bounds
    aux_checkFieldClass(options['nn'], 'reuse_bounds', ['bool'], 'options')
    
    # max_bounds
    aux_checkFieldClass(options['nn'], 'max_bounds', ['int', 'float'], 'options')
    
    # do_pre_order_reduction
    aux_checkFieldClass(options['nn'], 'do_pre_order_reduction', ['bool'], 'options')
    
    # max_gens_post
    aux_checkFieldClass(options['nn'], 'max_gens_post', ['int', 'float'], 'options')
    
    # remove_GI
    aux_checkFieldClass(options['nn'], 'remove_GI', ['bool'], 'options')
    
    # force_approx_lin_at
    aux_checkFieldClass(options['nn'], 'force_approx_lin_at', ['int', 'float'], 'options')
    
    # propagate_bounds
    aux_checkFieldClass(options['nn'], 'propagate_bounds', ['bool'], 'options')
    
    # sort_exponents
    aux_checkFieldClass(options['nn'], 'sort_exponents', ['bool'], 'options')
    
    # maxpool_type
    aux_checkFieldStr(options['nn'], 'maxpool_type', ['project', 'regression'], 'options')
    
    # order_reduction_sensitivity
    aux_checkFieldClass(options['nn'], 'order_reduction_sensitivity', ['bool'], 'options')

    # gnn ---
    
    # graph
    aux_checkFieldClass(options['nn'], 'graph', ['type(None)'], 'options')  # graph() equivalent

    # idx_pert_edges
    aux_checkFieldClass(options['nn'], 'idx_pert_edges', ['list', 'numpy.ndarray'], 'options')

    # invsqrt_order
    aux_checkFieldClass(options['nn'], 'invsqrt_order', ['int', 'float'], 'options')

    # invsqrt_use_approx_error
    aux_checkFieldClass(options['nn'], 'invsqrt_use_approx_error', ['bool'], 'options')

    # check training fields ---

    # use_approx_error
    aux_checkFieldClass(options['nn'], 'use_approx_error', ['bool'], 'options')

    if set_train_fields:
        aux_checkFieldClass(options['nn']['train'], 'optim', ['str'], 'options')
        # Note: nnSGDOptimizer, nnAdamOptimizer classes not yet implemented
        
        aux_checkFieldStr(options['nn']['train'], 'loss', ['mse', 'softmax+log', 'custom'], 'options')
        
        # Check warm_up vs ramp_up consistency
        if options['nn']['train']['warm_up'] > options['nn']['train']['ramp_up']:
            raise ValueError("options.nn.train.warm_up cannot be greater than options.nn.train.ramp_up")

        aux_checkFieldStr(options['nn']['train'], 'shuffle_data', ['never', 'every_epoch'], 'options')

        aux_checkFieldStr(options['nn']['train'], 'method', 
                         ['point', 'set', 'madry', 'gowal', 'trades', 'sabr', 'rand', 'extreme', 'naive', 'grad'], 'options')
        
        # 'set' training parameters
        aux_checkFieldStr(options['nn']['train'], 'volume_heuristic', ['interval', 'f-radius'], 'options')
        aux_checkFieldStr(options['nn']['train'], 'zonotope_weight_update', 
                         ['center', 'sample', 'extreme', 'outer_product', 'sum'], 'options')
        
        # Check use_approx_error vs num_approx_err consistency
        if (options['nn']['use_approx_error'] and (options['nn']['train']['num_approx_err'] == 0)) or \
           (not options['nn']['use_approx_error'] and options['nn']['train']['num_approx_err'] > 0):
            raise ValueError("options.nn.use_approx_error has to be true when options.nn.train.num_approx_err > 0")
        
        aux_checkFieldStr(options['nn']['train'], 'init_gens', 
                         ['l_inf', 'random', 'sensitivity', 'fgsm', 'patches'], 'options')
        
        # Regression poly method is not supported for training
        aux_checkFieldStr(options['nn'], 'poly_method', ['bounds', 'singh', 'center'], 'options')
    
    return options


# Auxiliary functions -----------------------------------------------------

def aux_checkFieldStr(options_nn: Dict[str, Any], field: str, admissible_values: list, struct_name: str):
    """
    Check if field has an admissible value.
    
    Args:
        options_nn: Options.nn dictionary
        field: Field name to check
        admissible_values: List of admissible values
        struct_name: Name of the structure for error messages
    """
    field_value = options_nn[field]
    if not isinstance(field_value, str) or field_value not in admissible_values:
        raise ValueError("Field {} has invalid value '{}'. Admissible values: {}".format(
            aux_getName(struct_name, field), field_value, admissible_values))


def aux_checkFieldClass(options_nn: Dict[str, Any], field: str, admissible_classes: list, struct_name: str):
    """
    Check if a field has the correct class.
    
    Args:
        options_nn: Options.nn dictionary
        field: Field name to check
        admissible_classes: List of admissible class types
        struct_name: Name of the structure for error messages
    """
    field_value = options_nn[field]
    field_type = type(field_value).__name__
    
    # Handle special case for None
    if field_value is None and 'type(None)' in admissible_classes:
        return
    
    # Check if the field type is in admissible classes
    valid_type = False
    for cls in admissible_classes:
        if cls == 'type(None)':
            if field_value is None:
                valid_type = True
                break
        elif cls == 'bool':
            if isinstance(field_value, bool):
                valid_type = True
                break
        elif cls == 'str':
            if isinstance(field_value, str):
                valid_type = True
                break
        elif cls == 'int':
            if isinstance(field_value, int):
                valid_type = True
                break
        elif cls == 'float':
            if isinstance(field_value, float):
                valid_type = True
                break
        elif cls == 'list':
            if isinstance(field_value, list):
                valid_type = True
                break
        elif cls == 'numpy.ndarray':
            if hasattr(field_value, 'shape'):  # Check if it's a numpy array
                valid_type = True
                break
        else:
            # Try to use isinstance for other types
            try:
                if isinstance(field_value, cls):
                    valid_type = True
                    break
            except TypeError:
                # Skip this type check if it's not a valid type
                continue
    
    if not valid_type:
        raise ValueError("Field {} has invalid type '{}'. Admissible types: {}".format(
            aux_getName(struct_name, field), field_type, admissible_classes))


def aux_getName(struct_name: str, field: str) -> str:
    """
    Get the full field name for error messages.
    
    Args:
        struct_name: Name of the structure
        field: Field name
        
    Returns:
        Full field name
    """
    return "{}.nn.{}".format(struct_name, field)


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
