"""
not_ - Logical NOT for polytopes

For convex sets, complement is generally non-convex and not representable
as a polytope in CORA API. Raise notSupported, mirroring MATLAB behavior.
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def not_(*args, **kwargs):
    raise CORAerror('CORA:notSupported', 'Logical NOT for polytopes is not supported')


