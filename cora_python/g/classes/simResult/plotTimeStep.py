"""
plotTimeStep - plots the time step size used in individual
    simulations over time (all simulations in one graph)

Syntax:
    han = plotTimeStep(simRes)
    han = plotTimeStep(simRes,type)

Inputs:
    simRes - simResult object
    type - (optional) plot settings (LineSpec and Name-Value pairs)

Outputs:
    han - handle to the graphics object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also:
"""

from typing import TYPE_CHECKING, Optional, Any
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .simResult import SimResult

def plotTimeStep(simRes: 'SimResult', *args, **kwargs) -> Optional[Any]:
    """
    Plots the time step size used in individual simulations over time.
    
    Args:
        simRes: simResult object
        *args: Additional plot arguments
        **kwargs: Additional plot keyword arguments
        
    Returns:
        Handle to the graphics object (matplotlib figure)
    """
    # Check if simRes is empty
    if simRes.isemptyobject():
        raise ValueError("simResult object is empty")
    
    # Handle single simResult or list of simResults
    simRes_list = simRes if isinstance(simRes, list) else [simRes]
    
    # Check hold status
    was_interactive = plt.isinteractive()
    if not was_interactive:
        plt.ion()
    
    # Get current axes or create new figure
    fig = plt.gcf()
    ax = plt.gca()
    
    # min / max for axis (if time is const, eps differences are shown...)
    mintimestep = np.inf
    maxtimestep = -np.inf
    cumsummin = np.inf
    cumsummax = -np.inf
    
    # loop over all simulations
    nrSim = len(simRes_list)
    
    for r in range(nrSim):
        simRes_r = simRes_list[r]
        
        for i in range(len(simRes_r.t)):
            # time axis - create step function representation
            t_vec = simRes_r.t[i]
            if len(t_vec) < 2:
                continue
                
            # Calculate time step sizes
            dt_vec = np.diff(t_vec)
            
            # Create step function for plotting
            # Each time step is repeated twice to create horizontal lines
            cumsumtVec = np.concatenate([[t_vec[0]], 
                                       np.repeat(t_vec[1:-1], 2), 
                                       [t_vec[-1]]])
            tVec = np.repeat(dt_vec, 2)
            
            # plot
            ax.plot(cumsumtVec, tVec, *args, **kwargs)
            
            # for axis limits
            if len(tVec) > 0:
                mintimestep = min(mintimestep, np.min(tVec))
                maxtimestep = max(maxtimestep, np.max(tVec))
                cumsummin = min(cumsummin, cumsumtVec[0])
                cumsummax = max(cumsummax, cumsumtVec[-1])
    
    # labels
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\Delta t$')
    
    # axes limits
    if np.isfinite(mintimestep) and np.isfinite(maxtimestep):
        ax.set_xlim([cumsummin, cumsummax])
        ax.set_ylim([0.9 * mintimestep, 1.1 * maxtimestep])
    
    # grid
    ax.grid(True)
    ax.set_box_aspect(None)
    
    # Restore interactive state
    if not was_interactive:
        plt.ioff()
    
    return fig 