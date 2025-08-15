"""
priv_copyProperties - copy cached set properties between polytopes

Copies properties like _emptySet_val, _bounded_val, _fullDim_val, _minHRep_val, _minVRep_val
from source to target if present. This mirrors MATLAB's helper.
"""

def priv_copyProperties(target, source):
    for attr in ['_emptySet_val', '_bounded_val', '_fullDim_val', '_minHRep_val', '_minVRep_val']:
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))
    return target


