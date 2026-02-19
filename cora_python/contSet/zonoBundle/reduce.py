"""
reduce - Reduces the order of a zonotope bundle

Syntax:
    zB = reduce(zB, option, order, filterLength)

Inputs:
    zB - zonotope bundle
    option - reduction method selector
    order - maximum order of reduced zonotope
    filterLength - optional filter length

Outputs:
    zB - bundle of reduced zonotopes

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/reduce

Authors:       Matthias Althoff (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       09-November-2010 (MATLAB)
Last update:   ---
Last revision: ---
"""


def reduce(zB, option, *args):
    from cora_python.contSet.zonoBundle import ZonoBundle

    zB_out = ZonoBundle(zB)
    for i in range(zB_out.parallelSets):
        zB_out.Z[i] = zB_out.Z[i].reduce(option, *args)
    return zB_out
