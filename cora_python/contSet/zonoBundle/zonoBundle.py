"""
zonoBundle - object constructor for zonotope bundles

Description:
    This class represents zonotope bundle defined as
    ∩_j=1^k {c_j + ∑_{i=1}^p_j beta_i * g_j^(i) | beta_i ∈ [-1,1]},
    i.e., the intersection of k zonotopes

Syntax:
    obj = zonoBundle(list)

Inputs:
    list - cell-array list = {Z1,Z2,...} storing the zonotopes that
           define the zonotope bundle

Outputs:
    obj - zonoBundle object

Example:
    Z1 = zonotope([1 3 0; 1 0 2])
    Z2 = zonotope([0 2 2; 0 2 -2])
    zB = zonoBundle({Z1,Z2})

    figure; hold on;
    plot(zB,[1,2],'FaceColor','r')
    plot(Z1,[1,2],'b')
    plot(Z2,[1,2],'g')

References:
    [1] M. Althoff. "Zonotope bundles for the efficient computation of 
        reachable sets", 2011

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       09-November-2010 (MATLAB)
Last update:   14-December-2022 (TL, property check in inputArgsCheck, MATLAB)
               29-March-2023 (TL, optimized constructor, MATLAB)
Last revision: 16-June-2023 (MW, restructure using auxiliary functions, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, List, Union

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros import CHECKS_ENABLED

if TYPE_CHECKING:
    from cora_python.contSet.zonotope.zonotope import Zonotope


class ZonoBundle(ContSet):
    """
    Class for representing zonotope bundles
    
    Properties (SetAccess = {?contSet, ?matrixSet}, GetAccess = public):
        Z: list of zonotopes
        parallelSets: number of zonotopes (internally-set property)
    """
    
    def __init__(self, *varargin):
        """
        Class constructor for zonotope bundles
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        assertNarginConstructor([1], len(varargin))

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], ZonoBundle):
            # Direct assignment like MATLAB
            other = varargin[0]
            self.Z = other.Z
            self.parallelSets = other.parallelSets
            super().__init__()
            self.precedence = 100
            return

        # 2. parse input arguments: varargin -> vars
        Z = _aux_parseInputArgs(*varargin)

        # 3. check correctness of input arguments
        _aux_checkInputArgs(Z, len(varargin))

        # 4. compute internal properties
        parallelSets = len(Z)

        # 5. assign properties
        self.Z = Z
        self.parallelSets = parallelSets

        # 6. set precedence (fixed) and initialize parent
        super().__init__()
        self.precedence = 100


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> List:
    """Parse input arguments from user and assign to variables"""
    
    # set default values
    Z = setDefaultValues([{}], list(varargin))[0]
    
    # Convert to list if needed
    if not isinstance(Z, list):
        Z = list(Z) if hasattr(Z, '__iter__') else [Z]
    
    return Z


def _aux_checkInputArgs(Z: List, n_in: int):
    """Check correctness of input arguments"""
    
    # only check if macro set to true
    if CHECKS_ENABLED and n_in > 0:

        inputArgsCheck([[Z, 'att', 'cell']])
    
        # check if zonotopes
        from cora_python.contSet.zonotope.zonotope import Zonotope
        if not all(isinstance(z, Zonotope) for z in Z):
            raise CORAerror('CORA:wrongInputInConstructor',
                          'First input argument has to be a list of zonotope objects.')
        
        # all zonotopes have to be of the same dimension
        from cora_python.contSet.zonotope.dim import dim
        dimensions = [dim(z) for z in Z]
        if len(set(dimensions)) > 1:
            raise CORAerror('CORA:wrongInputInConstructor',
                          'Zonotopes have to be embedded in the same affine space.') 