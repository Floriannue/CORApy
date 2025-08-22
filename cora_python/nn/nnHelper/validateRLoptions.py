"""
validateRLoptions - checks input and sets default values for the
   options.rl struct for the agentRL function.

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Lukas Koller
         Python: AI Assistant
"""

from typing import Dict, Any
from .setDefaultFields import setDefaultFields
from .validateNNoptions import validateNNoptions


def validateRLoptions(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check input and set default values for the options.rl struct for the agentRL function.
    
    Args:
        options: options dictionary (see agentRL)
        
    Returns:
        options: updated options
        
    See also: agentRL
    """
    # default values
    defaultRLfields = [
        ['gamma', 0.99],
        ['tau', 0.005],
        ['expNoise', 0.2],
        ['expNoiseTarget', 0.2],
        ['expNoiseType', 'OU'],
        ['expDecayFactor', 1],
        ['batchsize', 64],
        ['buffersize', 1e6],
        ['noise', 0.1],
        ['earlyStop', float('inf')],
        ['printFreq', 50],
        ['visRate', 50],
    ]
    
    # default fields for training the actor
    defaultActorTrainFields = [
        ['optim', lambda opts: {'type': 'Adam', 'lr': 1e-4, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'weight_decay': 0}],
        ['eta', 0.1],
        ['omega', 0.5],
        ['exact_backprop', False],
        ['zonotope_weight_update', 'outer_product'],
    ]
    
    # default fields for training the critic
    defaultCriticTrainFields = [
        ['optim', lambda opts: {'type': 'Adam', 'lr': 1e-3, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'weight_decay': 1e-2}],
        ['eta', 0.01],
        ['exact_backprop', False],
        ['zonotope_weight_update', 'outer_product'],
    ]
    
    # default fields for adv ops
    defaultAdvOps = [
        ['numSamples', 200],
        ['alpha', 4],
        ['beta', 4],
    ]
    
    # Validate neural network options. Do not set training options.
    options = validateNNoptions(options, False)
    
    # Set default RL options.
    if 'rl' not in options:
        options['rl'] = {}
    options['rl'] = setDefaultFields(options['rl'], defaultRLfields)
    
    # Set default actor training options.
    if 'actor' not in options['rl'] or 'nn' not in options['rl']['actor'] or 'train' not in options['rl']['actor']['nn']:
        if 'actor' not in options['rl']:
            options['rl']['actor'] = {}
        if 'nn' not in options['rl']['actor']:
            options['rl']['actor']['nn'] = {}
        if 'train' not in options['rl']['actor']['nn']:
            options['rl']['actor']['nn']['train'] = {}
    
    options['rl']['actor']['nn']['train'] = setDefaultFields(options['rl']['actor']['nn']['train'], defaultActorTrainFields)
    # Validate actor options. Set training options.
    options['rl']['actor'] = validateNNoptions(options['rl']['actor'], True)
    
    # Set default critic training options.
    if 'critic' not in options['rl'] or 'nn' not in options['rl']['critic'] or 'train' not in options['rl']['critic']['nn']:
        if 'critic' not in options['rl']:
            options['rl']['critic'] = {}
        if 'nn' not in options['rl']['critic']:
            options['rl']['critic']['nn'] = {}
        if 'train' not in options['rl']['critic']['nn']:
            options['rl']['critic']['nn']['train'] = {}
    
    options['rl']['critic']['nn']['train'] = setDefaultFields(options['rl']['critic']['nn']['train'], defaultCriticTrainFields)
    # Validate critic options. Set training options.
    options['rl']['critic'] = validateNNoptions(options['rl']['critic'], True)
    
    # Validate adversarial attack options.
    if 'advOps' not in options['rl']['actor']['nn']['train']:
        options['rl']['actor']['nn']['train']['advOps'] = {}
    options['rl']['actor']['nn']['train']['advOps'] = setDefaultFields(options['rl']['actor']['nn']['train']['advOps'], defaultAdvOps)
    
    # Check rl fields
    # Note: The validation checks from MATLAB are implemented here but adapted for Python
    # Some CORA-specific interval checks are simplified due to missing infrastructure
    
    # Check rl fields
    aux_checkFieldNumericDefInterval(options['rl'], 'gamma', (0, 1), 'options')
    aux_checkFieldNumericDefInterval(options['rl'], 'tau', (0, 1), 'options')
    aux_checkFieldNumericDefInterval(options['rl'], 'expNoise', (0, float('inf')), 'options')
    aux_checkFieldNumericDefInterval(options['rl'], 'expNoiseTarget', (0, float('inf')), 'options')
    aux_checkFieldStr(options['rl'], 'expNoiseType', ['OU', 'gaussian'], 'options')
    aux_checkFieldNumericDefInterval(options['rl'], 'expDecayFactor', (-1, 1), 'options')
    aux_checkFieldNumericDefInterval(options['rl'], 'batchsize', (0, float('inf')), 'options')
    aux_checkFieldNumericDefInterval(options['rl'], 'noise', (0, float('inf')), 'options')

    # Check actor fields
    aux_checkFieldNumericDefInterval(options['rl']['actor']['nn']['train'], 'eta', (0, float('inf')), 'options')
    aux_checkFieldNumericDefInterval(options['rl']['actor']['nn']['train'], 'omega', (0, 1), 'options')

    # Check critic fields
    aux_checkFieldNumericDefInterval(options['rl']['critic']['nn']['train'], 'eta', (0, float('inf')), 'options')

    # Validate the train options 
    if options['rl']['critic']['nn']['train']['method'] == 'set':
        if options['rl']['actor']['nn']['train']['method'] != 'set':
            raise ValueError("critic.nn.train.method cannot be 'set' when actor.nn.train.method is not 'set'")
    
    aux_checkFieldStr(options['rl']['critic']['nn']['train'], 'method', ['point', 'set'], 'options')
    
    return options


# Auxiliary functions -----------------------------------------------------

def aux_checkFieldStr(options_rl: Dict[str, Any], field: str, admissible_values: list, struct_name: str):
    """
    Check if field has an admissible value.
    
    Args:
        options_rl: Options.rl dictionary
        field: Field name to check
        admissible_values: List of admissible values
        struct_name: Name of the structure for error messages
    """
    field_value = options_rl[field]
    if not isinstance(field_value, str) or field_value not in admissible_values:
        raise ValueError("Field {} has invalid value '{}'. Admissible values: {}".format(
            aux_getName(struct_name, field), field_value, admissible_values))


def aux_checkFieldNumericDefInterval(options_rl: Dict[str, Any], field: str, interval_bounds: tuple, struct_name: str):
    """
    Check if a numeric field is within the specified interval bounds.
    
    Args:
        options_rl: Options.rl dictionary
        field: Field name to check
        interval_bounds: Tuple of (lower_bound, upper_bound)
        struct_name: Name of the structure for error messages
    """
    field_value = options_rl[field]
    lower_bound, upper_bound = interval_bounds
    
    if not isinstance(field_value, (int, float)) or field_value < lower_bound or field_value > upper_bound:
        raise ValueError("Field {} has value {} which is outside the valid domain [{}, {}]".format(
            aux_getName(struct_name, field), field_value, lower_bound, upper_bound))


def aux_getName(struct_name: str, field: str) -> str:
    """
    Get the full field name for error messages.
    
    Args:
        struct_name: Name of the structure
        field: Field name
        
    Returns:
        Full field name
    """
    return "{}.rl.{}".format(struct_name, field)
