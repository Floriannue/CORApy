<span style="display: inline-flex; align-items: flex-start;">
  <img src="coraLogo_readme.svg" alt="CORA" />
  <span style="font-size: 2.5em; margin-left: 0.1em; margin-top: -0.3em;">.py</span>
</span>

An supervised automatic translation of <a href='https://cora.in.tum.de' target='_blank'>CORA</a> from MATLAB to Python using Large Language Models (LLMs).

CORA is a tool for COntinuous Reachability Analysis.

<hr style="height: 1px;">

The Continuous Reachability Analyzer (CORA) is a MATLAB-based toolbox designed for the formal verification of cyber-physical systems through reachability analysis. 
It offers a comprehensive suite of tools for modeling and analyzing various system dynamics, including linear, nonlinear, and hybrid systems. 
CORA supports both continuous and discrete-time systems, accommodating uncertainties in system inputs and parameters. 
These uncertainties are captured by a diverse range of set representations such as intervals, zonotopes, Taylor models, and polytopes. 
Additionally, CORA provides functionalities for the formal verification of neural networks as well as data-driven system identification with reachset conformance. 
Various converters are implemented to easily model a system in CORA such as the well-established SpaceEx format for dynamic systems and ONNX format for neural networks. 
CORA ensures the seamless integration of different reachability algorithms without code modifications and aims for a user-friendly experience through automatic parameter tuning, 
making it a versatile tool for researchers and engineers in the field of cyber-physical systems.

### Reachability Analysis for Continuous Systems

CORA computes reachable sets for linear systems, nonlinear systems as well as for systems with constraints. 
Continuous as well as discrete time models are supported.  
Uncertainty in the system inputs as well as uncertainty in the model parameters can be explicitly considered. 
In addition, CORA also provides capabilities for the simulation of dynamical models.

### Reachability Analysis for Hybrid Systems

The toolbox is also capable of computing the reachable sets for hybrid systems. 
All implemented dynamic system classes can be used to describe the different continuous flows for the discrete system states. 
Further, multiple different methods for the computation of the intersections with guard sets are implemented in CORA.

### Geometric Sets

CORA has a modular design, making it possible to use the capabilities of the various set representations for other purposes besides reachability analysis. 
The toolbox implements vector set representation, e.g., intervals, zonotopes, Taylor models, and polytopes, 
as well as matrix set representations such as matrix zonotopes and interval matrices.

### Neural Network Verification and Robust Training

CORA enables the formal verification of neural networks, both in open-loop and in closed-loop scenarios. 
Open-loop verification refers to the task where properties of the output set of a neural network are verified, e.g., correctly classified images given noisy input. 
In closed-loop scenarios, the neural network is used as a controller of a dynamic system and is neatly integrated in the reachability algorithms above, e.g., controlling a car while keeping a safe distance. 
Additionally, one can train verifiably robust neural networks in CORA.


### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Floriannue/CORApy.git
cd CORApy

# 2. Create environment and install python dependencies
conda env create -f environment.yml
conda activate corapy
# alternative poetry install or pip install . without virtual environment

# 3. checkout the examples
python cora_python/examples/contDynamics/linearSys/example_linear_reach_01_5dim.py
python cora_python/examples/nn/example_neuralNetwork_verify_safe.py
python cora_python/examples/nn/example_neuralNetwork_verify_unsafe.py
```

### Folder Structure

| Folder | Description |
|---|---|
| `./contDynamics`| continuous dynamics classes |
| `./contSet`| continuous set classes and operations |
| `./converter`| converter from various formats to CORA |
| `./examples`| examples demonstrating the capabilities of CORA |
| `./g`| global helper classes, functions, and macros |
| `./hybridDynamics`| hybrid dynamics classes |
| `./matrixSet`| matrix set classes |
| `./nn`| neural network verification and robust training |
| `./specification`| specification classes for verification |
| `./tests`| unit tests |

<hr style="height: 1px;">