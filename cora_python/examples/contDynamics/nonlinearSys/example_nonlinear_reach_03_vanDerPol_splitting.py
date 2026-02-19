"""
example_nonlinear_reach_03_vanDerPol_splitting - example of nonlinear
   reachability analysis with splitting, can be found in [1, Sec. 3.4.5]
   or in [2]. A new technique for computing this example with less
   splitting has been published in [3].

Syntax:
    completed = example_nonlinear_reach_03_vanDerPol_splitting()

Inputs:
    -

Outputs:
    completed - true/false

References:
    [1] M. Althoff, "Reachability analysis and its application to the
        safety assessment of autonomous cars", Dissertation, TUM 2010
    [2] M. Althoff et al. "Reachability analysis of nonlinear systems with
        uncertain parameters using conservative linearization", CDC 2008
    [3] M. Althoff. "Reachability analysis of nonlinear systems using
        conservative polynomialization and non-convex sets", HSCC 2013

Authors:       Matthias Althoff (MATLAB)
Written:       26-June-2009
Last update:   23-April-2020 (restructure params/options)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025

================================================================================
HOW TO RUN THE MATLAB VERSION (truth reference)
================================================================================
1. Open MATLAB and go to the CORA root (folder containing contDynamics, contSet,
   global, models, etc.). If using this repo, that is the cora_matlab folder:
     cd('d:\Bachelorarbeit\Translate_Cora\cora_matlab')
2. Add CORA to the path (if not already):
     addpath(genpath('.'))
   Or from the repo root:
     addpath(genpath('d:\Bachelorarbeit\Translate_Cora\cora_matlab'))
3. Run the example (the .m file defines a function; you must call it):
     example_nonlinear_reach_03_vanDerPol_splitting()
   From repo root, first add path then call:
     addpath(genpath('cora_matlab'))
     example_nonlinear_reach_03_vanDerPol_splitting()
   Or via batch from repo root:
     matlab -batch "addpath(genpath('cora_matlab')); example_nonlinear_reach_03_vanDerPol_splitting()"

================================================================================
TRANSLATION FIDELITY (Python vs MATLAB)
================================================================================
- Structure and flow: 1:1 (params, options, nonlinearSys, reach, simulateRandom,
  plot, labels). Indexing: MATLAB 1-based -> Python 0-based (e.g. projDim [1 2] -> [0 1]).
- Initial set: MATLAB uses zonoBundle(Z0) for exact splitting; Python uses a single
  Zonotope with the same center and generators as the first (and only) zonotope
  in that bundle. Reason: zonoBundle + zonotope in linReach is not yet implemented
  in Python, so R0 is Zonotope(c,G) with c=[1.4;2.3], G=[0.3 0; 0 0.05].
- Output set: MATLAB uses default compOutputSet; Python sets compOutputSet=False
  because nonlinearSys without an output equation has no out_jacobian in Python.
  Effect: initial time-point set is the state set (no output mapping), same as
  MATLAB when no output equation is defined.
- Dynamics: vanderPolEq(x,u) matches MATLAB (sympy branch for derivatives, numpy
  for simulation). System: NonlinearSys(vanderPolEq, states=2, inputs=1) matches
  nonlinearSys(@vanderPolEq) (dimensions inferred in MATLAB from function handle).
================================================================================

Python run (from repo root, PowerShell):
  cd "d:\Bachelorarbeit\Translate_Cora"
  $env:PYTHONPATH = "d:\Bachelorarbeit\Translate_Cora"
  python cora_python/examples/contDynamics/nonlinearSys/example_nonlinear_reach_03_vanDerPol_splitting.py
"""

import sys
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Add the project root to the path so imports work when run as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, project_root)

# Configure matplotlib for smooth rendering like MATLAB
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.antialiased'] = True
plt.rcParams['patch.antialiased'] = True
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.8

from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.models.Cora.vanDerPol.vanderPolEq import vanderPolEq


def example_nonlinear_reach_03_vanDerPol_splitting():
    """
    Example of nonlinear reachability analysis with splitting (van der Pol).
    Computes reachable set, runs random simulations, and plots results.
    """
    # Parameters --------------------------------------------------------------
    # MATLAB: params.tFinal = 0.5;
    # MATLAB: Z0{1} = zonotope([1.4 0.3 0; 2.3 0 0.05]); params.R0 = zonoBundle(Z0);
    # Zonotope(c, G): c = [1.4; 2.3], G = [0.3 0; 0 0.05]
    params = {
        'tFinal': 0.5,
        'R0': ZonoBundle([Zonotope(np.array([[1.4], [2.3]]), np.array([[0.3, 0], [0, 0.05]]))])
    }

    # Reachability Settings ---------------------------------------------------
    # MATLAB: options.timeStep = 0.02; options.taylorTerms = 4; etc.
    options = {
        'timeStep': 0.02,
        'taylorTerms': 4,
        'zonotopeOrder': 10,
        'intermediateOrder': 10,
        'errorOrder': 5,
        'alg': 'lin',
        'tensorOrder': 3,
        'maxError': 0.05 * np.ones((2, 1)),
        'reductionInterval': 100,
        'verbose': True,
        'compOutputSet': False  # no output equation in van der Pol dynamics
    }

    # System Dynamics ---------------------------------------------------------
    # MATLAB: vanderPol = nonlinearSys(@vanderPolEq);
    vanderPol = NonlinearSys(vanderPolEq, states=2, inputs=1)

    # Reachability Analysis ---------------------------------------------------
    print('Computing reachable set...')
    timer_val = time.perf_counter()
    R = vanderPol.reach(params, options)
    t_comp = time.perf_counter() - timer_val
    print(f'Computation time of reachable set: {t_comp:.4f}')

    # Handle R as list (multiple branches) or single ReachSet
    if isinstance(R, list):
        R_plot = R
    else:
        R_plot = [R]

    # Simulation --------------------------------------------------------------
    # MATLAB: simOpt.points = 60; traj = simulateRandom(vanderPol, params, simOpt);
    sim_opt = {'points': 60}
    traj = vanderPol.simulateRandom(params, sim_opt)

    # Visualization -----------------------------------------------------------
    # MATLAB: projDim = [1 2]; -> Python 0-based: [0, 1]
    proj_dim = [0, 1]
    plt.figure()
    plt.gca().set_aspect('auto')
    plt.hold = True if hasattr(plt, 'hold') else None

    # plot reachable sets
    for idx, r_branch in enumerate(R_plot):
        if idx == 0:
            r_branch.plot(proj_dim, DisplayName='Reachable set')
        else:
            r_branch.plot(proj_dim)

    # plot initial set (MATLAB: R(1).R0 -> first branch)
    R_plot[0].R0.plot(proj_dim, DisplayName='Initial set')

    # plot simulation results
    traj.plot(proj_dim, DisplayName='Simulations')

    # label plot (MATLAB: xlabel(['x_{',num2str(projDim(1)),'}']); 1-based)
    plt.xlabel(f'$x_{{{proj_dim[0] + 1}}}$')
    plt.ylabel(f'$x_{{{proj_dim[1] + 1}}}$')
    plt.box(True)
    plt.legend()
    plt.tight_layout()

    # Save figure (examples must run; saving allows verification without GUI)
    out_dir = os.path.join(project_root, 'cora_python', 'examples', 'contDynamics', 'nonlinearSys')
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'example_nonlinear_reach_03_vanDerPol_splitting.png')
    plt.savefig(fig_path)
    print(f'Saved plot to {fig_path}')

    plt.show()
    plt.close()

    # example completed
    completed = True
    print('Example completed successfully.')
    return completed


if __name__ == '__main__':
    example_nonlinear_reach_03_vanDerPol_splitting()
