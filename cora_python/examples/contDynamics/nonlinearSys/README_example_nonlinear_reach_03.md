# example_nonlinear_reach_03_vanDerPol_splitting

Nonlinear reachability with splitting (van der Pol oscillator) and plotting.

## Run MATLAB (truth reference)

From the **repository root** `Translate_Cora`:

```matlab
addpath(genpath('cora_matlab'));
example_nonlinear_reach_03_vanDerPol_splitting();
```

Or from the CORA root (e.g. `cora_matlab`):

```matlab
cd('cora_matlab')   % or full path, e.g. d:\Bachelorarbeit\Translate_Cora\cora_matlab
addpath(genpath('.'));
example_nonlinear_reach_03_vanDerPol_splitting();
```

Batch (e.g. from PowerShell, repo root):

```powershell
matlab -batch "addpath(genpath('cora_matlab')); example_nonlinear_reach_03_vanDerPol_splitting()"
```

## Run Python

From the **repository root**, with CORA on `PYTHONPATH`:

```powershell
cd "d:\Bachelorarbeit\Translate_Cora"
$env:PYTHONPATH = "d:\Bachelorarbeit\Translate_Cora"
python cora_python/examples/contDynamics/nonlinearSys/example_nonlinear_reach_03_vanDerPol_splitting.py
```

## Translation fidelity

See the docstring at the top of `example_nonlinear_reach_03_vanDerPol_splitting.py` for a detailed comparison. In short:

- **Same:** params (tFinal, R0 shape), options (timeStep, taylorTerms, alg, etc.), flow (reach → simulateRandom → plot), dynamics (`vanderPolEq`), plot axes and labels.
- **Differences:** Python uses a single `Zonotope` for R0 (same c, G as MATLAB’s first zonotope in the bundle) because `zonoBundle + zonotope` in `linReach` is not yet implemented; Python sets `compOutputSet=False` when there is no output equation so no `out_jacobian` is required.
