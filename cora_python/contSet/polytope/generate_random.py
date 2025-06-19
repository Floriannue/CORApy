# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

import numpy as np
from typing import Union, Tuple, List, Any
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.check import checkNameValuePairs
from cora_python.g.functions.matlab.validate.preprocessing import readNameValuePair
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Auxiliary functions (will be implemented in separate private files)
from cora_python.contSet.polytope.private.aux_check_consistency import aux_check_consistency
from cora_python.contSet.polytope.private.aux_set_default_values import aux_set_default_values
from cora_python.contSet.polytope.private.aux_1d_unbounded_nondeg import aux_1d_unbounded_nondeg
from cora_python.contSet.polytope.private.aux_2d_unbounded_nondeg import aux_2d_unbounded_nondeg
from cora_python.contSet.polytope.private.aux_2d_unbounded_deg import aux_2d_unbounded_deg
from cora_python.contSet.polytope.private.aux_nd_unbounded import aux_nd_unbounded
from cora_python.contSet.polytope.private.aux_1d_bounded_deg import aux_1d_bounded_deg
from cora_python.contSet.polytope.private.aux_nd_bounded import aux_nd_bounded

def generate_random(*varargin) -> Polytope:
    """
    generateRandom - Generates a random non-empty polytope
       restrictions:
       - number of constraints >= 1    =>  can be unbounded
       - number of constraints >= 2    =>  + can be unbounded and degenerate
       - number of constraints >= n+1  =>  + can be bounded
       - number of constraints >= n+2  =>  + can be bounded and degenerate

    Syntax:
       P_out = Polytope.generate_random()
       P_out = Polytope.generate_random('Dimension',n)
       P_out = Polytope.generate_random('Dimension',n,'NrConstraints',nrCon)
       P_out = Polytope.generate_random('Dimension',n,'NrConstraints',nrCon,...
              'IsDegenerate',isDeg)
       P_out = Polytope.generate_random('Dimension',n,'NrConstraints',nrCon,...
              'IsDegenerate',isDeg,'IsBounded',isBounded)

    Inputs:
       Name-Value pairs (all options, arbitrary order):
          <'Dimension',n> - dimension
          <'NrConstraints',nrCon> - number of constraints
          <'IsDegenerate',isDeg> - degeneracy (true/false)
          <'IsBounded',isBounded> - boundedness (true/false)

    Outputs:
       P_out - random polytope

    Example: 
       P = Polytope.generate_random('Dimension',2);
       # plot(P); # Plotting is not implemented yet

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: zonotope/generateRandom

    Authors:       Niklas Kochdumper, Mark Wetzlinger
    Written:       05-May-2020
    Last update:   10-November-2022 (MW, adapt to new name-value pair syntax)
                   30-November-2022 (MW, new algorithm)
                   12-December-2022 (MW, less randomness for more stability)
    Last revision: 14-July-2024 (MW, refactor)
    """

    # name-value pairs -> number of input arguments is always a multiple of 2
    if len(varargin) % 2 != 0:
        raise CORAerror('CORA:evenNumberInputArgs')
    else:
        # read input arguments
        NVpairs = list(varargin)
        # check list of name-value pairs
        checkNameValuePairs(NVpairs, ['Dimension', 'NrConstraints', 'IsDegenerate', 'IsBounded'])
        
        # dimension given?
        NVpairs, n = readNameValuePair(NVpairs, 'Dimension', lambda x: isinstance(x, (int, np.integer)) and x > 0)
        # number of constraints given?
        NVpairs, nr_con = readNameValuePair(NVpairs, 'NrConstraints', lambda x: isinstance(x, (int, np.integer)) and x > 0)
        # degeneracy given?
        NVpairs, is_deg = readNameValuePair(NVpairs, 'IsDegenerate', lambda x: isinstance(x, bool))
        # boundedness given?
        NVpairs, is_bnd = readNameValuePair(NVpairs, 'IsBounded', lambda x: isinstance(x, bool))

    # quick exits (errors)
    aux_check_consistency(n, nr_con, is_deg, is_bnd)

    # set default values: depends on which parameters are given and their values
    n, nr_con, is_deg, is_bnd = aux_set_default_values(n, nr_con, is_deg, is_bnd)

    A, b, Ae, be = np.array([]), np.array([]), np.array([]), np.array([])

    if not is_bnd:
        # idea for unbounded case: sample directions

        # degenerate case: one pair of constaints needs to be of the form
        #    a*x <= b   a*x >= b

        if n == 1 and not is_deg:
            A, b, Ae, be = aux_1d_unbounded_nondeg(n, nr_con)
        elif n == 2 and not is_deg:
            A, b, Ae, be = aux_2d_unbounded_nondeg(n, nr_con)
        elif n == 2 and is_deg:
            A, b, Ae, be = aux_2d_unbounded_deg(n, nr_con)
        else:
            A, b, Ae, be = aux_nd_unbounded(n, nr_con, is_deg)
        
    else:
        # bounded case: generate simplex and add constraints
        # it is ensured that nrCon > n+1 (non-degenerate)
        #                    nrCon > n+2 (degenerate)

        if n == 1 and is_deg:
            A, b, Ae, be = aux_1d_bounded_deg(n, nr_con)
        else:
            A, b, Ae, be = aux_nd_bounded(n, nr_con, is_deg)

    # instantiate polytope
    P_out = Polytope(A, b, Ae, be)

    # set properties
    P_out._bounded.val = is_bnd
    P_out._full_dim.val = not is_deg
    P_out._empty_set.val = False

    return P_out 