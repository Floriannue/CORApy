# CORApy - CORA Translation Project

Bachelor project: supervised automatic translation of the [CORA toolbox](https://github.com/TUMcps/CORA) from MATLAB to Python using Large Language Models (LLMs). The process is organized in [Cursor](https://www.cursor.com), with **Claude-4-Sonnet** as the primary model. readme_florian2.md contains my lates instructions for the model. In the archive folder are some of the past debug_scripts, readmes and plots.

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Floriannue/Translate_Cora.git
cd Translate_Cora

# 2. Initialize and update the CORA submodule
# Only do so if you need CORA - the cora_python translation does not require it to run.
git submodule init
git submodule update --remote --recursive
#Follow the install instructions in cora_matlab/README.md if you want to run cora_matlab
#The translation process can then run matlab tests

# 3. Install Python dependencies using Poetry (recommended)
#pip install poetry if not available
poetry install #(or pip install .)

# 4. run example
python cora_python/examples/contDynamics/linearSys/example_linear_reach_01_5dim.py
python cora_python/examples/nn/example_neuralNetwork_verify_safe.py
python cora_python/examples/nn/example_neuralNetwork_verify_unsafe.py
```