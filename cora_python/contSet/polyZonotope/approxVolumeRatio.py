"""
approxVolumeRatio - Calculates the approximate ratio of the volumes 
   between the polynomial zonotope constructed by only the dependent
   generators of the given polynomial zonotope and the zonotope
   constructed by the independent generator part of the polynomial
   zonotope; the ratio is computed as ratio = (V_ind/V_dep)^(1/n)

Syntax:
    ratio = approxVolumeRatio(pZ)
    ratio = approxVolumeRatio(pZ,type)

Inputs:
    pZ - polyZonotope object
    type - method used to calculate the volume ('interval' or 'pca')

Outputs:
    ratio - approximate volume ratio (V_ind/V_dep)^(1/n)

Example:
    pZ = polyZonotope([0;0],[1 0 1;1 2 -2],[-1 0.3;1.2 0.2],[1 0 1;0 1 2]);
    ratio = approxVolumeRatio(pZ)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/volume, interval/volume

Authors:       Niklas Kochdumper
Written:       25-July-2018 
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


def approxVolumeRatio(pZ: 'PolyZonotope', *varargin) -> float:
    """
    Calculates the approximate ratio of the volumes between the polynomial 
    zonotope constructed by only the dependent generators and the zonotope
    constructed by the independent generator part
    
    Args:
        pZ: polyZonotope object
        *varargin: optional type argument ('interval' or 'pca')
        
    Returns:
        ratio: approximate volume ratio (V_ind/V_dep)^(1/n)
    """
    
    # parse input arguments
    # MATLAB: type = setDefaultValues({'interval'},varargin);
    type_val = setDefaultValues(['interval'], list(varargin))
    
    # check input arguments
    # MATLAB: inputArgsCheck({{pZ,'att','polyZonotope'};
    #                         {type,'str',{'interval','pca'}}});
    inputArgsCheck([(pZ, 'att', 'polyZonotope'),
                    (type_val, 'str', ['interval', 'pca'])])
    
    # special cases
    # MATLAB: if isempty(pZ.GI)
    if pZ.GI.size == 0 or (pZ.GI.ndim > 0 and pZ.GI.shape[1] == 0):
        ratio = 0.0
        return ratio
    
    # MATLAB: if isempty(pZ.G)
    if pZ.G.size == 0 or (pZ.G.ndim > 0 and pZ.G.shape[1] == 0):
        ratio = np.inf
        return ratio
    
    # MATLAB: if strcmp(type,'pca')
    if type_val == 'pca':
        # calculate state-space-transformation with pca
        # MATLAB: G = [pZ.G -pZ.G pZ.GI -pZ.GI];
        G = np.hstack([pZ.G, -pZ.G, pZ.GI, -pZ.GI])
        
        # MATLAB: [T,~,~] = svd(G);
        from scipy.linalg import svd
        T, _, _ = svd(G, full_matrices=False)
        
        # transform the polynomial zonotope to the new state space
        # MATLAB: pZ = T'*pZ;
        pZ = T.T @ pZ
    
    # over-approximate the independent generators part with an interval
    # MATLAB: n = dim(pZ);
    n = pZ.dim()
    
    # MATLAB: Z = zonotope([zeros(n,1),pZ.GI]);
    Z = Zonotope(np.zeros((n, 1)), pZ.GI)
    
    # MATLAB: I_ind = interval(Z);
    I_ind = Z.interval()
    
    # over-approximate the dependent generators part with an interval
    # MATLAB: pZ.GI = zeros(n,0);
    # Create a copy to avoid modifying the original
    import copy
    pZ_copy = copy.deepcopy(pZ)
    pZ_copy.GI = np.zeros((n, 0))
    
    # MATLAB: Idep = interval(pZ);
    Idep = pZ_copy.interval()
    
    # remove dimensions that are all-zero
    # MATLAB: ind1 = find(rad(Idep) == 0);
    # MATLAB: ind2 = find(rad(I_ind) == 0);
    rad_Idep = Idep.rad()
    rad_I_ind = I_ind.rad()
    
    ind1 = np.where(rad_Idep == 0)[0]
    ind2 = np.where(rad_I_ind == 0)[0]
    
    # MATLAB: ind = unique([ind1;ind2]);
    # MATLAB: ind = setdiff(1:length(Idep),ind);
    ind_remove = np.unique(np.concatenate([ind1, ind2]))
    ind = np.setdiff1d(np.arange(len(Idep)), ind_remove)
    
    # calculate the volumes of the parallelotopes
    # MATLAB: Vind = volume_(I_ind(ind));
    # MATLAB: Vdep = volume_(Idep(ind));
    if len(ind) == 0:
        # If all dimensions are zero, ratio is undefined, return 0
        ratio = 0.0
        return ratio
    
    I_ind_sub = I_ind[ind] if hasattr(I_ind, '__getitem__') else I_ind
    Idep_sub = Idep[ind] if hasattr(Idep, '__getitem__') else Idep
    
    Vind = I_ind_sub.volume_()
    Vdep = Idep_sub.volume_()
    
    # calculate the volume ratio
    # MATLAB: ratio = (Vind/Vdep)^(1/n);
    if Vdep == 0:
        ratio = np.inf
    else:
        ratio = (Vind / Vdep) ** (1.0 / n)
    
    return ratio

