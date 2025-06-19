# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from typing import Tuple, List
from cora_python.contSet.polytope.polytope import Polytope

def get_print_set_info(P: Polytope) -> Tuple[str, List[str]]:
    """
    getPrintSetInfo - returns all information to properly print a set 
    to the command window 

    Syntax:
       [abbrev,propertyOrder] = get_print_set_info(P)

    Inputs:
       P - polytope

    Outputs:
       abbrev - set abbreviation
       propertyOrder - order of the properties

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: none

    Authors:       Tobias Ladner
    Written:       10-October-2024
    Last update:   ---
    Last revision: ---
    """

    abbrev = 'P'
    if P._is_h_rep.val:
        property_order = ['A', 'b', 'Ae', 'be']
    else:  # is_v_rep
        property_order = ['V']

    return abbrev, property_order 