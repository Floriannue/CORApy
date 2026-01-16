"""
priv_reduceAdaptive - reduces the zonotope order until a maximum amount
    of over-approximation defined by the Hausdorff distance between the
    original zonotope and the reduced zonotope; based on [Thm 3.2,1]

Syntax:
    pZ = priv_reduceAdaptive(pZ,diagpercent)

Inputs:
    pZ - polyZonotope object
    diagpercent - percentage of diagonal of box over-approximation of
                  polyZonotope (used to compute dHmax)

Outputs:
    pZ - reduced polyZonotope object

Example: 
    c = [0;0];
    G = [2 0 1 0.02 0.003; 0 2 1 0.01 -0.001];
    GI = [0;0.5];
    E = [1 0 3 0 1;0 1 1 2 1];
    pZ = polyZonotope(c,G,GI,E);
    pZ = reduce(pZ,'adaptive',0.05);

References:
    [1] Wetzlinger et al. "Adaptive Parameter Tuning for Reachability 
        Analysis of Nonlinear Systems", HSCC 2021             

Other m-files required: none
Subfunctions: see below
MAT-files required: none

See also: reduce

Authors:       Mark Wetzlinger
Written:       01-October-2020
Last update:   16-June-2021 (restructure input/output args)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope


def priv_reduceAdaptive(pZ: 'PolyZonotope', diagpercent: float) -> 'PolyZonotope':
    """
    Adaptive reduction for polynomial zonotopes
    
    Args:
        pZ: polyZonotope object
        diagpercent: percentage of diagonal of box over-approximation
        
    Returns:
        pZ: reduced polyZonotope object
    """
    # read data from pZ
    G = pZ.G
    GI = pZ.GI
    E = pZ.E
    
    # dep. gens with only-even exponents
    # MATLAB: halfs = ~any(mod(E,2),1);
    if E.size > 0:
        halfs = ~np.any(E % 2, axis=0)
        if np.any(halfs):
            # decrease G if only-even exponents
            # MATLAB: G(:,halfs) = G(:,halfs) * 0.5;
            G = G.copy()  # Make a copy to avoid modifying original
            G[:, halfs] = G[:, halfs] * 0.5
    else:
        halfs = np.array([], dtype=bool)
    
    n, nrG = G.shape if G.size > 0 else (pZ.dim(), 0)
    nrGI = GI.shape[1] if GI.size > 0 else 0
    Gabs = np.abs(G) if G.size > 0 else np.zeros((n, 0))
    GIabs = np.abs(GI) if GI.size > 0 else np.zeros((n, 0))
    
    # set dHmax percentage of diagonal of box(Z)
    # MATLAB: Gbox = sum([Gabs,GIabs],2); % same as: rad(interval(pZ))
    if Gabs.size > 0 and GIabs.size > 0:
        Gbox = np.sum(np.hstack([Gabs, GIabs]), axis=1, keepdims=True)
    elif Gabs.size > 0:
        Gbox = np.sum(Gabs, axis=1, keepdims=True)
    elif GIabs.size > 0:
        Gbox = np.sum(GIabs, axis=1, keepdims=True)
    else:
        Gbox = np.zeros((n, 1))
    
    # MATLAB: dHmax = (diagpercent * 0.5) * sqrt(sum(Gbox.^2));
    dHmax = (diagpercent * 0.5) * np.sqrt(np.sum(Gbox**2))
    
    m_dep = np.array([])
    hadapG = np.array([])
    if Gabs.size > 0:
        # order generators for selection by extended girard metrics
        # dep. gens: similar to vanilla reduce
        # MATLAB: m_dep = sum(Gabs,1);
        m_dep = np.sum(Gabs, axis=0)
        # MATLAB: [~,idxDep] = mink(m_dep,nrG);
        # mink returns smallest k values, but we want all sorted
        idxDep = np.argsort(m_dep)  # ascending order (smallest first)
        
        # compute dH over-approximation for dep. gens
        # use pZ.G to account for center shift
        # (we use max distance between origin and symm.int.hull)
        # multiplicative factor 2 needed due to deletion of dep. gen from G
        # MATLAB: hadapG = 0.5 * vecnorm(cumsum(abs(pZ.G(:,idxDep)),2),2);
        pZ_G_abs = np.abs(pZ.G[:, idxDep])
        cumsum_G = np.cumsum(pZ_G_abs, axis=1)
        # vecnorm computes 2-norm along dimension 2 (columns), which is axis=0 in Python
        hadapG = 0.5 * np.linalg.norm(cumsum_G, axis=0)
        # disect individual cumulative sum to unify below to combined metrics
        # MATLAB: hadapG = [hadapG(1),diff(hadapG)];
        hadapG = np.concatenate([[hadapG[0]], np.diff(hadapG)])
    
    m_indep = np.array([])
    hadapGI = np.array([])
    if GIabs.size > 0:
        # indep. gens: same as for zonotopes
        # MATLAB: [norminf,maxidx] = max(GIabs,[],1);
        norminf = np.max(GIabs, axis=0)
        maxidx = np.argmax(GIabs, axis=0)
        # MATLAB: normsum = sum(GIabs,1);
        normsum = np.sum(GIabs, axis=0)
        # MATLAB: m_indep = normsum - norminf;
        m_indep = normsum - norminf
        # MATLAB: [~,idxIndep] = mink(m_indep,nrGI);
        idxIndep = np.argsort(m_indep)  # ascending order
        
        # compute additional vector for indep. gens, use linear indexing
        # MATLAB: muGIabs = zeros(n,nrGI);
        # MATLAB: muGIabs(n*(0:nrGI-1)+maxidx) = norminf;
        muGIabs = np.zeros((n, nrGI))
        for i in range(nrGI):
            muGIabs[maxidx[i], i] = norminf[i]
        
        # compute new over-approximation of dH
        # MATLAB: GIdiag = cumsum(GIabs(:,idxIndep)-muGIabs(:,idxIndep),2);
        GIabs_sorted = GIabs[:, idxIndep]
        muGIabs_sorted = muGIabs[:, idxIndep]
        GIdiag = np.cumsum(GIabs_sorted - muGIabs_sorted, axis=1)
        # MATLAB: hadapGI = 2 * vecnorm(GIdiag,2);
        hadapGI = 2 * np.linalg.norm(GIdiag, axis=0)
        # disect individual cumulative sus to unify below to combined metrics
        # MATLAB: hadapGI = [hadapGI(1),diff(hadapGI)];
        hadapGI = np.concatenate([[hadapGI[0]], np.diff(hadapGI)])
    
    # order for both together
    # MATLAB: [~,idxall] = mink([m_dep,m_indep],nrG + nrGI);
    m_all = np.concatenate([m_dep, m_indep]) if m_dep.size > 0 and m_indep.size > 0 else (m_dep if m_dep.size > 0 else m_indep)
    idxall = np.argsort(m_all)  # ascending order
    
    hext = np.zeros(nrG + nrGI)
    # MATLAB: hext(idxall > nrG) = hadapGI;
    # MATLAB: hext(idxall <= nrG) = hadapG;
    idxall_gt_nrG = idxall >= nrG  # 0-based: > nrG becomes >= nrG
    idxall_le_nrG = idxall < nrG   # 0-based: <= nrG becomes < nrG
    
    if hadapGI.size > 0:
        hext[idxall_gt_nrG] = hadapGI[idxall[idxall_gt_nrG] - nrG]
    if hadapG.size > 0:
        hext[idxall_le_nrG] = hadapG[idxall[idxall_le_nrG]]
    
    # MATLAB: hext = cumsum(hext);
    hext = np.cumsum(hext)
    
    # reduce until over-approximation of dH hits dHmax
    # MATLAB: redUntil = find(hext <= dHmax,1,'last');
    redUntil_idx = np.where(hext <= dHmax)[0]
    if redUntil_idx.size > 0:
        redUntil = redUntil_idx[-1] + 1  # +1 because MATLAB find returns 1-based, we need count
    else:
        redUntil = 0
    
    # MATLAB: idxDep = idxall(idxall(1:redUntil) <= nrG);
    idxall_selected = idxall[:redUntil]
    idxDep = idxall_selected[idxall_selected < nrG]
    # MATLAB: idxIndep = idxall(idxall(1:redUntil) > nrG) - nrG;
    idxIndep = idxall_selected[idxall_selected >= nrG] - nrG
    
    # delete converted generators
    if len(idxDep) > 0:
        # MATLAB: pZ.G(:,idxDep) = [];
        keep_dep = np.setdiff1d(np.arange(nrG), idxDep)
        if len(keep_dep) > 0:
            pZ.G = pZ.G[:, keep_dep]
            if E.size > 0:
                E = E[:, keep_dep]
        else:
            pZ.G = np.zeros((n, 0))
            E = np.zeros((0, 0), dtype=int)
    
    if len(idxIndep) > 0:
        # MATLAB: GI(:,idxIndep) = [];
        keep_ind = np.setdiff1d(np.arange(nrGI), idxIndep)
        if len(keep_ind) > 0:
            GI = GI[:, keep_ind]
        else:
            GI = np.zeros((n, 0))
    
    # MATLAB: if isempty(idxDep) && isempty(idxIndep)
    if len(idxDep) == 0 and len(idxIndep) == 0:
        pZ.GI = GI
    else:
        # MATLAB: pZ.GI = [GI, diag(sum([Gabs(:,idxDep),GIabs(:,idxIndep)],2))];
        Gabs_idxDep = Gabs[:, idxDep] if len(idxDep) > 0 else np.zeros((n, 0))
        GIabs_idxIndep = GIabs[:, idxIndep] if len(idxIndep) > 0 else np.zeros((n, 0))
        if Gabs_idxDep.size > 0 and GIabs_idxIndep.size > 0:
            diag_sum = np.diag(np.sum(np.hstack([Gabs_idxDep, GIabs_idxIndep]), axis=1))
        elif Gabs_idxDep.size > 0:
            diag_sum = np.diag(np.sum(Gabs_idxDep, axis=1))
        elif GIabs_idxIndep.size > 0:
            diag_sum = np.diag(np.sum(GIabs_idxIndep, axis=1))
        else:
            diag_sum = np.zeros((n, 0))
        
        if GI.size > 0 and diag_sum.size > 0:
            pZ.GI = np.hstack([GI, diag_sum])
        elif GI.size > 0:
            pZ.GI = GI
        elif diag_sum.size > 0:
            pZ.GI = diag_sum
        else:
            pZ.GI = np.zeros((n, 0))
    
    # shift pZ.c by center of zonotope converted from dep. gens
    if np.any(halfs) and len(idxDep) > 0:
        # MATLAB: temp = find(halfs);
        temp = np.where(halfs)[0]
        # MATLAB: temp = temp(ismember(temp,idxDep));
        temp = temp[np.isin(temp, idxDep)]
        if len(temp) > 0:
            # MATLAB: pZ.c = pZ.c + sum(G(:,temp),2); % 0.5 factor already done above
            pZ.c = pZ.c + np.sum(G[:, temp], axis=1, keepdims=True)
    
    # remove all unused dependent factors (empty rows in E)
    # MATLAB: temp = any(E,2);
    if E.size > 0:
        temp = np.any(E, axis=1)
        pZ.E = E[temp, :]
        if pZ.id.size > 0:
            pZ.id = pZ.id[temp]
    else:
        pZ.E = np.zeros((0, 0), dtype=int)
        pZ.id = np.zeros((0, 1), dtype=int)
    
    return pZ
