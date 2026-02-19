"""
copy - copies the zonoBundle object (used for dynamic dispatch)

Syntax:
    zB_out = copy(zB)

Inputs:
    zB - zonoBundle object

Outputs:
    zB_out - copied zonoBundle object

Example:
    Z1 = zonotope([1;1],[1 1; -1 1]);
    Z2 = zonotope([-1;1],[1 0; 0 1]);
    zB = zonoBundle({Z1,Z2});
    zB_out = copy(zB);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       30-September-2024 (MATLAB)
Last update:   ---
Last revision: ---
"""


def copy(zB):
    from cora_python.contSet.zonoBundle import ZonoBundle
    return ZonoBundle(zB)
