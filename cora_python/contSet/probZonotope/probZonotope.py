"""
probZonotope - class for probabilistic zonotopes

Syntax:
    obj = probZonotope(Z,G)
    obj = probZonotope(Z,G,gamma)

Inputs:
    Z - zonotope matrix Z = [c,g1,...,gp]
    G - matrix storing the probabilistic generators G = [g1_, ..., gp_]
    gamma - cut-off value for plotting. The set is cut-off at 2*sigma,
            where sigma is the variance

Outputs:
    obj - Generated object

Example:
    Z = [10 1 -2; 0 1 1]
    G = [0.6 1.2; 0.6 -1.2]
    probZ1 = probZonotope(Z,G)
    gamma = 1.5
    probZ2 = probZonotope(Z,G,gamma)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval,  polytope

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       03-August-2007 (MATLAB)
Last update:   26-February-2008 (MATLAB)
               20-March-2015 (MATLAB)
               04-May-2020 (MW, transition to classdef, MATLAB)
               14-December-2022 (TL, property check in inputArgsCheck, MATLAB)
Last revision: 16-June-2023 (MW, restructure using auxiliary functions, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Union, Optional

from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.macros import CHECKS_ENABLED

if TYPE_CHECKING:
    pass


class ProbZonotope(ContSet):
    """
    Class for representing probabilistic zonotopes
    
    Properties (SetAccess = protected, GetAccess = public):
        Z: zonotope matrix
        g: probabilistic generators
        cov: covariance matrix
        gauss: determining if cov is updated (internally-set property)
        gamma: cut-off mSigma value (internally-set property)
    """
    
    def __init__(self, *varargin):
        """
        Class constructor for probabilistic zonotopes
        """
        # 0. avoid empty instantiation
        if len(varargin) == 0:
            raise CORAerror('CORA:noInputInSetConstructor')
        assertNarginConstructor(list(range(1, 4)), len(varargin))

        # 1. copy constructor
        if len(varargin) == 1 and isinstance(varargin[0], ProbZonotope):
            # Direct assignment like MATLAB
            other = varargin[0]
            self.Z = other.Z
            self.g = other.g
            self.gamma = other.gamma
            self.gauss = other.gauss
            self.cov = other.cov
            super().__init__()
            return

        # 2. parse input arguments: varargin -> vars
        Z, g, gamma = _aux_parseInputArgs(*varargin)

        # 3. check correctness of input arguments
        _aux_checkInputArgs(Z, g, gamma, len(varargin))

        # 4. assign properties
        self.Z = Z
        self.g = g
        self.gamma = gamma
        self.gauss = False

        # 5. compute cov and initialize parent
        super().__init__()
        self.cov = self.sigma()
        self.gauss = True

    def __repr__(self):
        return f"ProbZonotope(Z={self.Z.shape}, g={self.g.shape}, gamma={self.gamma})"


# Auxiliary functions -----------------------------------------------------

def _aux_parseInputArgs(*varargin) -> Tuple[np.ndarray, np.ndarray, float]:
    """Parse input arguments from user and assign to variables"""
    
    # no input arguments
    if len(varargin) == 0:
        Z = np.array([])
        g = np.array([])
        gamma = 2
        return Z, g, gamma

    # set default values
    Z, g, gamma = setDefaultValues([[], [], 2], list(varargin))

    # Convert to numpy arrays
    Z = np.array(Z) if Z is not None else np.array([])
    g = np.array(g) if g is not None else np.array([])
    gamma = float(gamma) if gamma is not None else 2

    return Z, g, gamma


def _aux_checkInputArgs(Z: np.ndarray, g: np.ndarray, gamma: float, n_in: int):
    """Check correctness of input arguments"""
    
    # only check if macro set to true
    if CHECKS_ENABLED and n_in > 0:

        inputChecks = [
            [Z, 'att', 'numeric', 'finite'],
            [g, 'att', 'numeric', 'finite'],
            [gamma, 'att', 'numeric']
        ]
        
        inputArgsCheck(inputChecks) 